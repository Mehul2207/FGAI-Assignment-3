"""Fine-tune google/mt5-small (encoder-decoder) for En->Hi translation."""
import time

import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

from common import *

MODEL = "google/mt5-small"
PREFIX = "translate English to Hindi: "
MAXLEN = CFG["mt5_len"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    train, val, test = load_split("train", CFG["mt5_n"]), load_split("val", 200), load_split("test")

    def prep(batch):
        enc = tok([PREFIX + s for s in batch["src"]], max_length=MAXLEN, truncation=True)
        enc["labels"] = tok(text_target=batch["tgt"], max_length=MAXLEN, truncation=True)["input_ids"]
        return enc

    cols = ["src", "tgt"]
    tr = Dataset.from_list(train).map(prep, batched=True, remove_columns=cols)
    va = Dataset.from_list(val).map(prep, batched=True, remove_columns=cols)

    args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/mt5", per_device_train_batch_size=CFG["mt5_bs"], per_device_eval_batch_size=16,
        learning_rate=5e-4, num_train_epochs=CFG["mt5_epochs"], weight_decay=0.01, warmup_ratio=0.05,
        eval_strategy="epoch", save_strategy="no", logging_steps=200, report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # mT5 is unstable in fp16
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model))
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    model.to(dev).eval()

    @torch.no_grad()
    def translate(sents, temperature=0.0, bs=32):
        out = []
        for i in range(0, len(sents), bs):
            enc = tok([PREFIX + s for s in sents[i:i + bs]], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAXLEN).to(dev)
            kw = dict(do_sample=True, temperature=temperature, top_k=0) if temperature > 0 else dict(do_sample=False)
            ids = model.generate(**enc, max_new_tokens=MAXLEN, num_beams=1, **kw)
            out += tok.batch_decode(ids, skip_special_tokens=True)
        return out

    srcs, refs = [r["src"] for r in test], [r["tgt"] for r in test]
    t1 = time.time()
    preds = translate(srcs)
    infer_time = time.time() - t1
    n = count_params(model)
    info = {"model": "mT5-small", "arch": "encoder-decoder", "method": "full fine-tuning",
            "params": n, "trainable_params": n, "train_time_s": train_time,
            "infer_time_s": infer_time, "n_test": len(test), **score(preds, refs)}
    info["temp_sweep"] = temp_sweep(translate, srcs, refs)
    save_result("mt5", info, preds)


if __name__ == "__main__":
    main()
