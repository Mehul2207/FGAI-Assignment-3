"""Aggregate results: comparison table, plots, side-by-side samples and simple responsible-AI flags."""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RES_DIR, load_split

files = sorted(f for f in glob.glob(f"{RES_DIR}/*.json"))
results = {os.path.basename(f)[:-5]: json.load(open(f)) for f in files}
rows = []
for k, r in results.items():
    rows.append({"run": k, "model": r["model"], "arch": r["arch"], "method": r["method"],
                 "params (M)": round(r["params"] / 1e6, 1),
                 "trainable (M)": round(r["trainable_params"] / 1e6, 2),
                 "BLEU": round(r["bleu"], 2), "chrF": round(r["chrf"], 2),
                 "train time (min)": round(r["train_time_s"] / 60, 1),
                 "infer time (s)": round(r["infer_time_s"], 1),
                 "ms/sentence": round(1000 * r["infer_time_s"] / r["n_test"], 1)})
df = pd.DataFrame(rows)
df.to_csv(f"{RES_DIR}/comparison.csv", index=False)
open(f"{RES_DIR}/comparison.md", "w").write(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False))
print(df.to_string(index=False))

# --- bar plots ---
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, col in zip(ax, ["BLEU", "chrF", "ms/sentence"]):
    a.bar(df["run"], df[col]); a.set_title(col); a.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.savefig(f"{RES_DIR}/metrics.png", dpi=150); plt.close()

# --- temperature analysis ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, r in results.items():
    ts = sorted(r["temp_sweep"], key=float)
    for a, m in zip(ax, ["bleu", "chrf"]):
        a.plot([float(t) for t in ts], [r["temp_sweep"][t][m] for t in ts], marker="o", label=k)
for a, m in zip(ax, ["BLEU", "chrF"]):
    a.set_xlabel("sampling temperature"); a.set_ylabel(m); a.set_title(f"{m} vs temperature (200 test sents)")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{RES_DIR}/temperature.png", dpi=150); plt.close()
pd.DataFrame({k: {t: round(v["bleu"], 2) for t, v in r["temp_sweep"].items()} for k, r in results.items()}).to_csv(f"{RES_DIR}/temperature_bleu.csv")

# --- samples + simple automated flags for the ethics discussion ---
test = load_split("test")
preds = {k: open(f"{RES_DIR}/{k}_preds.txt", encoding="utf-8").read().split("\n") for k in results}
with open(f"{RES_DIR}/samples.md", "w", encoding="utf-8") as f:
    for i in range(min(25, len(test))):
        f.write(f"### {i}\n**EN:** {test[i]['src']}\n\n**REF:** {test[i]['tgt']}\n\n")
        for k in preds:
            f.write(f"- **{k}:** {preds[k][i]}\n")
        f.write("\n")

deva = re.compile(r"[\u0900-\u097F]")
latin = re.compile(r"[A-Za-z]")
flags = []
for k, p in preds.items():
    n = len(p)
    ratios = [len(p[i].split()) / max(1, len(test[i]["tgt"].split())) for i in range(n)]
    flags.append({"run": k,
                  "empty %": round(100 * sum(1 for x in p if not x.strip()) / n, 1),
                  "mostly-Latin (untranslated) %": round(100 * sum(1 for x in p if len(latin.findall(x)) > len(deva.findall(x))) / n, 1),
                  "len ratio >1.8 (possible hallucination) %": round(100 * sum(r > 1.8 for r in ratios) / n, 1),
                  "len ratio <0.4 (possible omission) %": round(100 * sum(r < 0.4 for r in ratios) / n, 1)})
pd.DataFrame(flags).to_csv(f"{RES_DIR}/responsible_ai_flags.csv", index=False)
print(pd.DataFrame(flags).to_string(index=False))
