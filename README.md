# English→Hindi Machine Translation: Transformer vs mT5 vs LLaMA vs Gemma

Assignment 3 – Foundations of Generative AI (ICT4403)

## 1. Problem statement and objective
Build and compare four neural machine-translation systems for **English → Hindi**:
encoder–decoder (scratch Transformer, mT5) vs decoder-only (LLaMA, Gemma), evaluated with BLEU, chrF,
training time, inference time and parameter count, plus a responsible-AI analysis.

## 2. Dataset
IITB English–Hindi parallel corpus – https://huggingface.co/datasets/cfilt/iitb-english-hindi
- Filtered to 3–50 words per side, length ratio 0.5–2.
- Subsets used: **50,000 train**, validation (official split, filtered), **1,000 test** (official test split, filtered).
- LLM LoRA fine-tuning uses the first 10,000 training pairs. All four models are scored on the *same* test set.
- (Fill in: exact counts printed by `python common.py`.)

## 3. Models
| Model | Architecture | Adaptation |
|---|---|---|
| Transformer (3+3 layers, d=256) | encoder–decoder | trained from scratch, joint 16k BPE |
| google/mt5-small | encoder–decoder | full fine-tuning |
| meta-llama/Llama-3.2-1B-Instruct | decoder-only | zero-shot, 3-shot, LoRA |
| google/gemma-3-1b-it | decoder-only | zero-shot, 3-shot, LoRA |

All use **greedy decoding** for the main comparison (comparable settings).

## 4. Execution steps
```bash
pip install -r requirements.txt
huggingface-cli login                      # Llama & Gemma are gated: accept licences first
python common.py                           # download + split data
python train_transformer.py
python finetune_mt5.py
for m in meta-llama/Llama-3.2-1B-Instruct google/gemma-3-1b-it; do
  for mode in zeroshot fewshot lora; do python decoder_only_mt.py --model $m --mode $mode; done
done
python evaluate.py                         # table, plots, samples, flags -> results/
```
GPU needed (Colab/Kaggle T4 works; use `--dtype fp32` for Gemma on GPUs without bf16 if you see NaNs).

### Laptop ("lite") vs GPU ("full") preset
Default is `lite` (10k train pairs, 200 test sentences, tiny Transformer, mT5 on 4k pairs for 1 epoch,
LoRA on 1k pairs, temperature sweep on 50 sentences at 0/0.7/1.0). For the bigger setup run with
`PRESET=full` (Linux/Mac) or `set PRESET=full` (Windows cmd). If you switch presets, delete `data/` and `checkpoints/` first.
Lite LLM runs used: Llama-3.2-1B-Instruct and gemma-3-270m-it, modes `zeroshot` and `lora`.

## 5. Results
_Paste `results/comparison.md` here._  Plots: `results/metrics.png`.

## 6. Model comparison and evaluation
_Discuss: model size, pretraining (scratch vs multilingual pretrained), tokenization (Hindi tokens-per-word in
Llama/Gemma vs mT5 vs custom BPE), adaptation method (full FT vs LoRA vs prompting), training cost, latency._

## 7. Sampling-temperature analysis
Temperatures 0.0, 0.3, 0.7, 1.0 on the first 200 test sentences: `results/temperature.png`, `results/temperature_bleu.csv`.
_Discuss how BLEU/chrF fall as temperature rises and which model is most robust._

## 8. Responsible and ethical AI
Samples: `results/samples.md`; automated flags: `results/responsible_ai_flags.csv`.
_Discuss bias (gendered pronouns/professions), hallucination, cultural misrepresentation, sensitive content;
mitigation: human review, confidence/QE filtering, glossaries, bias test sets, disclosure, content filters, feedback loop._

## 9. Conclusion
_State the best-performing model (by BLEU/chrF) and the best quality-per-compute trade-off._
