"""Shared utilities: data preparation, loading, metrics, result saving."""
import json
import os
import random

import sacrebleu
from datasets import load_dataset

DATA_DIR = "data"
RES_DIR = "results"
DATASET = "cfilt/iitb-english-hindi"   # https://huggingface.co/datasets/cfilt/iitb-english-hindi
SRC_LANG, TGT_LANG = "en", "hi"
PRESET = os.environ.get("PRESET", "lite")   # "lite" (laptop) or "full" (GPU)
CFG = {
    "lite": dict(n_train=10000, n_test=200, max_words=30, pool=60000, sweep_n=50, temps=[0.0, 0.7, 1.0],
                 tf_vocab=8000, tf_d=128, tf_layers=2, tf_ff=512, tf_epochs=15, tf_warm=500, tf_bs=64,
                 mt5_n=4000, mt5_epochs=1, mt5_bs=8, mt5_len=64,
                 llm_n=1000, llm_len=128, llm_bs=2, llm_new=96, llm_gen_bs=8),
    "full": dict(n_train=50000, n_test=1000, max_words=50, pool=400000, sweep_n=200, temps=[0.0, 0.3, 0.7, 1.0],
                 tf_vocab=16000, tf_d=256, tf_layers=3, tf_ff=1024, tf_epochs=20, tf_warm=2000, tf_bs=64,
                 mt5_n=50000, mt5_epochs=3, mt5_bs=16, mt5_len=128,
                 llm_n=10000, llm_len=256, llm_bs=4, llm_new=192, llm_gen_bs=16),
}[PRESET]
TEMPS = CFG["temps"]                   # sampling-temperature analysis
SWEEP_N = CFG["sweep_n"]               # sentences used for each temperature run


def prepare_data(n_train=CFG["n_train"], n_test=CFG["n_test"], max_words=CFG["max_words"], seed=42):
    """Download IITB En-Hi, filter, subsample and write train/val/test jsonl."""
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset(DATASET)

    def ok(ex):
        e, h = ex["translation"][SRC_LANG].strip(), ex["translation"][TGT_LANG].strip()
        le, lh = len(e.split()), len(h.split())
        return 3 <= le <= max_words and 3 <= lh <= max_words and 0.5 <= le / lh <= 2.0

    def dump(split, name, limit):
        rows = [{"src": ex["translation"][SRC_LANG].strip(),
                 "tgt": ex["translation"][TGT_LANG].strip()} for ex in split]
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
        with open(f"{DATA_DIR}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))

    train = ds["train"].shuffle(seed=seed).select(range(min(CFG["pool"], len(ds["train"])))).filter(ok)
    dump(train, "train", n_train)
    dump(ds["validation"].filter(ok), "val", 10_000)
    dump(ds["test"].filter(ok), "test", n_test)


def load_split(name, limit=None):
    with open(f"{DATA_DIR}/{name}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return rows[:limit] if limit else rows


def score(hyps, refs):
    return {"bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
            "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score}


def temp_sweep(gen_fn, srcs, refs, temps=TEMPS, n=SWEEP_N):
    """gen_fn(sentences, temperature) -> list[str]. Returns {temp: {bleu, chrf}}."""
    out = {}
    for t in temps:
        hyps = gen_fn(srcs[:n], t)
        out[str(t)] = score(hyps, refs[:n])
        print(f"temperature={t}: {out[str(t)]}")
    return out


def save_result(name, info, preds):
    os.makedirs(RES_DIR, exist_ok=True)
    with open(f"{RES_DIR}/{name}.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(f"{RES_DIR}/{name}_preds.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(p.replace("\n", " ") for p in preds))
    print("saved", name, {k: v for k, v in info.items() if k != "temp_sweep"})


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    prepare_data()
