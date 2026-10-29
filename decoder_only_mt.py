"""Decoder-only MT (LLaMA / Gemma): zero-shot, few-shot prompting, or LoRA fine-tuning.

Examples:
  python decoder_only_mt.py --model meta-llama/Llama-3.2-1B-Instruct --mode zeroshot
  python decoder_only_mt.py --model meta-llama/Llama-3.2-1B-Instruct --mode lora
  python decoder_only_mt.py --model google/gemma-3-1b-it --mode lora
Both models are gated on Hugging Face: accept the licence and run `huggingface-cli login` first.
"""
import argparse
import time

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import *

dev = "cuda" if torch.cuda.is_available() else "cpu"
HEAD = "Translate the following English sentence into Hindi.\n"


def zero_prompt(s):
    return f"{HEAD}English: {s}\nHindi:"


def make_few_prompt(shots):
    ctx = "".join(f"English: {r['src']}\nHindi: {r['tgt']}\n\n" for r in shots)
    return lambda s: f"{HEAD}\n{ctx}English: {s}\nHindi:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["zeroshot", "fewshot", "lora"], default="lora")
    ap.add_argument("--n_ft", type=int, default=CFG["llm_n"], help="training sentences for LoRA")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--max_len", type=int, default=CFG["llm_len"])
    ap.add_argument("--bs", type=int, default=CFG["llm_bs"])
    ap.add_argument("--accum", type=int, default=4)
    a = ap.parse_args()

    if a.dtype == "auto":
        a.dtype = "bf16" if (dev == "cuda" and torch.cuda.is_bf16_supported()) else ("fp16" if dev == "cuda" else "fp32")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    short = a.model.split("/")[-1].lower()
    name = f"{short}_{a.mode}"

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt).to(dev)
    base_params = count_params(model)

    train, test = load_split("train"), load_split("test")
    srcs, refs = [r["src"] for r in test], [r["tgt"] for r in test]
    prompt_fn = zero_prompt
    if a.mode == "fewshot":
        prompt_fn = make_few_prompt(train[:3])  # same 3 fixed examples for every test sentence

    train_time, trainable = 0.0, 0
    if a.mode == "lora":
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.print_trainable_parameters()

        def encode(r):
            p = tok(zero_prompt(r["src"]), add_special_tokens=True)["input_ids"]
            t = tok(" " + r["tgt"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (p + t)[: a.max_len]
            lab = ([-100] * len(p) + t)[: a.max_len]
            return torch.tensor(ids), torch.tensor(lab)

        data = [encode(r) for r in train[: a.n_ft]]
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
        steps = a.epochs * (len(data) // (a.bs * a.accum))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / (0.05 * steps + 1)) * max(0.0, 1 - s / steps))
        scaler = torch.amp.GradScaler("cuda", enabled=(a.dtype == "fp16"))
        model.train()
        t0, step = time.time(), 0
        for ep in range(a.epochs):
            perm = torch.randperm(len(data)).tolist()
            for i, s in enumerate(tqdm(range(0, len(perm), a.bs), desc=f"epoch {ep+1}")):
                b = [data[j] for j in perm[s:s + a.bs]]
                ids = pad_sequence([x[0] for x in b], batch_first=True, padding_value=tok.pad_token_id).to(dev)
                lab = pad_sequence([x[1] for x in b], batch_first=True, padding_value=-100).to(dev)
                att = (ids != tok.pad_token_id).long()
                att[:, 0] = 1
                with torch.autocast(dev, dtype=dt, enabled=(dev == "cuda" and a.dtype != "fp32")):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.accum
                scaler.scale(loss).backward()
                if (i + 1) % a.accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    sched.step()
                    step += 1
                    if step % 50 == 0:
                        print(f"step {step}/{steps} loss {loss.item() * a.accum:.3f}")
        train_time = time.time() - t0
        model.eval()
    model.eval()

    @torch.no_grad()
    def translate(sents, temperature=0.0, bs=CFG["llm_gen_bs"]):
        out = []
        for i in range(0, len(sents), bs):
            enc = tok([prompt_fn(s) for s in sents[i:i + bs]], return_tensors="pt", padding=True).to(dev)
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else dict(do_sample=False)
            gen = model.generate(**enc, max_new_tokens=CFG["llm_new"], pad_token_id=tok.pad_token_id, **kw)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [(t.strip().split("\n")[0] if t.strip() else "") for t in texts]  # keep first line only
        return out

    t1 = time.time()
    preds = translate(srcs)
    infer_time = time.time() - t1
    method = {"zeroshot": "zero-shot prompting", "fewshot": "3-shot prompting", "lora": f"LoRA fine-tuning ({a.n_ft} sents)"}[a.mode]
    info = {"model": f"{short} ({a.mode})", "arch": "decoder-only", "method": method,
            "params": base_params, "trainable_params": trainable or 0, "train_time_s": train_time,
            "infer_time_s": infer_time, "n_test": len(test), **score(preds, refs)}
    info["temp_sweep"] = temp_sweep(translate, srcs, refs)
    save_result(name, info, preds)


if __name__ == "__main__":
    main()
