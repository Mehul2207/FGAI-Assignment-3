"""Basic encoder-decoder Transformer trained from scratch (PyTorch) with a joint SentencePiece BPE vocab."""
import math
import os
import time

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from common import *

PAD, UNK, BOS, EOS = 0, 1, 2, 3
VOCAB, MAXLEN, EPOCHS, BS = CFG["tf_vocab"], 100, CFG["tf_epochs"], CFG["tf_bs"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


class PosEnc(nn.Module):
    def __init__(self, d, maxlen=256):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TransformerMT(nn.Module):
    def __init__(self, V, d=256, h=4, L=3, ff=1024, drop=0.1):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.pos = PosEnc(d)
        self.drop = nn.Dropout(drop)
        self.tf = nn.Transformer(d, h, L, L, ff, drop, batch_first=True)
        self.out = nn.Linear(d, V, bias=False)
        self.out.weight = self.emb.weight  # tied embeddings (joint vocab)

    def embed(self, x):
        return self.drop(self.pos(self.emb(x) * math.sqrt(self.d)))

    def encode(self, src):
        m = src == PAD
        return self.tf.encoder(self.embed(src), src_key_padding_mask=m), m

    def decode(self, tgt, mem, mmask):
        T = tgt.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt.device), 1)
        h = self.tf.decoder(self.embed(tgt), mem, tgt_mask=causal,
                            tgt_key_padding_mask=(tgt == PAD), memory_key_padding_mask=mmask)
        return self.out(h)

    def forward(self, src, tgt_in):
        mem, mm = self.encode(src)
        return self.decode(tgt_in, mem, mm)


def batches(sp, rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle:
        torch.manual_seed(int(time.time()))
        idx = torch.randperm(len(rows)).tolist()
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        s = [torch.tensor(sp.encode(r["src"])[:MAXLEN] + [EOS]) for r in chunk]
        t = [torch.tensor(sp.encode(r["tgt"])[:MAXLEN]) for r in chunk]
        src = pad_sequence(s, batch_first=True, padding_value=PAD)
        tin = pad_sequence([torch.cat([torch.tensor([BOS]), x]) for x in t], batch_first=True, padding_value=PAD)
        tout = pad_sequence([torch.cat([x, torch.tensor([EOS])]) for x in t], batch_first=True, padding_value=PAD)
        yield src.to(dev), tin.to(dev), tout.to(dev)


@torch.no_grad()
def translate(model, sp, sents, temperature=0.0, bs=64, maxlen=100):
    model.eval()
    outs = []
    for i in range(0, len(sents), bs):
        b = sents[i:i + bs]
        src = pad_sequence([torch.tensor(sp.encode(s)[:MAXLEN] + [EOS]) for s in b],
                           batch_first=True, padding_value=PAD).to(dev)
        mem, mm = model.encode(src)
        ys = torch.full((len(b), 1), BOS, device=dev)
        done = torch.zeros(len(b), dtype=torch.bool, device=dev)
        for _ in range(maxlen):
            logits = model.decode(ys, mem, mm)[:, -1]
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        for row in ys[:, 1:].tolist():
            row = [t for t in row if t != PAD]
            if EOS in row:
                row = row[:row.index(EOS)]
            outs.append(sp.decode(row))
    return outs


def main():
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists("data/spm.model"):
        with open("data/spm_train.txt", "w", encoding="utf-8") as f:
            for r in train:
                f.write(r["src"] + "\n" + r["tgt"] + "\n")
        spm.SentencePieceTrainer.train(input="data/spm_train.txt", model_prefix="data/spm", vocab_size=VOCAB,
                                       model_type="bpe", character_coverage=1.0,
                                       pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS)
    sp = spm.SentencePieceProcessor(model_file="data/spm.model")

    model = TransformerMT(VOCAB, d=CFG["tf_d"], L=CFG["tf_layers"], ff=CFG["tf_ff"]).to(dev)
    n_params = count_params(model)
    print("parameters:", n_params)
    d, warm = CFG["tf_d"], CFG["tf_warm"]
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * d ** -0.5 * min((s + 1) ** -0.5, (s + 1) * warm ** -1.5))
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    best, t0 = 1e9, time.time()
    for ep in range(EPOCHS):
        model.train()
        for src, tin, tout in tqdm(batches(sp, train, BS, True), desc=f"epoch {ep+1}"):
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = crit(model(src, tin).flatten(0, 1), tout.flatten())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(s, ti).flatten(0, 1), to.flatten()).item()
                     for s, ti, to in batches(sp, val, 128, False)) / math.ceil(len(val) / 128)
        print(f"epoch {ep+1} val_loss {vl:.3f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), "checkpoints/transformer.pt")
    train_time = time.time() - t0

    model.load_state_dict(torch.load("checkpoints/transformer.pt"))
    srcs, refs = [r["src"] for r in test], [r["tgt"] for r in test]
    t1 = time.time()
    preds = translate(model, sp, srcs)
    infer_time = time.time() - t1
    info = {"model": "Transformer (scratch)", "arch": "encoder-decoder", "method": "train from scratch",
            "params": n_params, "trainable_params": n_params, "train_time_s": train_time,
            "infer_time_s": infer_time, "n_test": len(test), **score(preds, refs)}
    info["temp_sweep"] = temp_sweep(lambda s, t: translate(model, sp, s, temperature=t), srcs, refs)
    save_result("transformer", info, preds)


if __name__ == "__main__":
    main()
