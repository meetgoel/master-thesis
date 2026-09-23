# Master Thesis — Cross-Lingual and Few-Shot Offensive Language Detection in Low-Resource Dravidian Languages: An Empirical Study of XLM-R and Large Language Models

Code and data for the MSc thesis *"Cross-Lingual Transfer versus In-Context Learning for Offensive Language Detection in Low-Resource Dravidian Languages: A Comparative Study of XLM-RoBERTa and Qwen2.5-1.5B-Instruct under Constrained Data Budgets."*

This repository contains the full experimental pipeline used to answer four research questions:

- **RQ1** — How effectively does a multilingual XLM-R model trained on a source-language dataset transfer to Kannada, Tamil, and Malayalam without target-language training data?
- **RQ2** — How does increasing the amount of labelled target-language data from 50 to 100 and 500 examples affect XLM-R performance relative to zero-shot transfer across Kannada, Tamil, and Malayalam?
- **RQ3** — Under comparable few-shot data budgets, how does XLM-R fine-tuning compare with Qwen2.5-1.5B in-context learning for offensive-language detection in Kannada, Tamil, and Malayalam?
- **RQ4** — How sensitive are few-shot XLM-R results to random seed variation, and does the observed data-efficiency trend remain consistent across different seeds?

The full write-up, with every number in this repository's output traced back to a section of the thesis, is in the accompanying dissertation document.

## Repository structure

```
master-thesis/
├── HateSpeech_v10.ipynb        # Full pipeline: data loading -> preprocessing -> all experiments
├── Datasets/
│   ├── HASOC 2019/
│   │   ├── Train Data/HASOC_2019_train.tsv
│   │   └── Test Data/HASOC_2019_test.tsv
│   ├── HASOC 2020 EN-DE-HI/
│   │   ├── Train Data/hasoc_2020_{en,de,hi}_train.xlsx
│   │   └── Test Data/{english,german,hindi}_test_1509.csv
│   ├── HASOC 2021/
│   │   ├── Train Data/{en_Hasoc2021_train.csv, train.tsv}
│   │   └── Test Data/{test_task1.csv, test.tsv}
│   └── DravidianCodeMix/
│       ├── kannada_offensive_full.csv
│       ├── tamil_offensive_full.tsv
│       └── malayalam_offensive_full.tsv
└── README.md
```

Raw dataset files are included exactly as distributed by their original sources (HASOC shared task organisers; Chakravarthi et al., 2022 — DravidianCodeMix). See **Data sources & licensing** below before reusing them.

## What the notebook does

The notebook is organised top-to-bottom in the order it must be executed:

1. **Setup** — clones this repo, mounts Google Drive for checkpoint persistence, installs dependencies.
2. **Data loading & EDA** — loads all HASOC and DravidianCodeMix files, inspects class balance, nulls, duplicates.
3. **Preprocessing** — text cleaning (URL/mention/emoji stripping), deduplication, and label harmonisation of every dataset onto a common binary `HOF`/`NOT` schema.
4. **Splitting** — a fixed, seeded 80/10/10 stratified train/val/test split per target language, reused by every experiment below.
5. **Baseline** — TF-IDF + Logistic Regression per target language.
6. **RQ1 — Zero-shot** — fine-tunes XLM-RoBERTa-base on a language-balanced English/German/Hindi HASOC pool, evaluates it directly on each target language's test set with no target-language training.
7. **RQ2 — Few-shot XLM-R** — continues fine-tuning the source model on stratified 50/100/500-example target-language samples, repeated for 3 random seeds per language/shot-size (27 runs total).
8. **RQ3 — Qwen2.5-1.5B-Instruct ICL** — prompts Qwen with the same 50/100-shot demonstration sets used in RQ2, under greedy decoding, with no gradient updates.
9. **RQ4 — Seed robustness** — aggregates the 27 RQ2 runs into per-condition mean ± standard deviation.
10. **LIME explainability** — compares zero-shot vs. 500-shot XLM-R feature attributions per language.
11. **Monolingual upper bound** — trains XLM-R from scratch on each language's full training split, for reference.
12. **Results consolidation** — builds the final comparison tables and figures used in the thesis.

## Reproducing the results

1. Open `HateSpeech_v10.ipynb` in Google Colab (a GPU runtime is required for the fine-tuning and LLM-inference cells).
2. Run the first two setup cells (repo clone, Drive mount) — Drive mounting is a **hard requirement**; the notebook will raise an error if it fails, since every checkpoint and result is persisted there.
3. Run the notebook top to bottom. Every stochastic stage (few-shot sampling, model fine-tuning, ICL demonstration sampling) checks for and reuses a cached result before recomputing, so a full re-run reproduces the exact figures reported in the thesis rather than generating new random samples.
4. Expect a full from-scratch run to take several hours on a single Colab GPU, dominated by the 27 XLM-R few-shot fine-tuning runs and the Qwen2.5 ICL inference sweep (up to 4,169 generation calls per condition for Tamil).

### Environment

Core dependencies (installed by the notebook's own `!pip install` cells):

```
transformers
datasets
evaluate
scikit-learn
torch
lime
pandas
numpy
matplotlib
seaborn
plotly
```

No pinned `requirements.txt` is included; the notebook was developed against whatever `transformers`/`datasets`/`torch` versions were current on Google Colab at the time of writing. If reproducing on a different platform, install recent versions of the packages above.

### Random seeds

- Data splitting: seed `42` (fixed, not varied).
- Source-language balancing: seed `42`.
- Few-shot sampling and fine-tuning (RQ2/RQ4): seeds `42`, `123`, `456`.
- ICL demonstration sampling (RQ3): seed `42` only.

## Key results (summary)

Full results, per-seed breakdowns, and discussion are in the thesis. Headline Macro-F1 by language and condition:

| Condition | Kannada | Tamil | Malayalam |
|---|---|---|---|
| TF-IDF + LR baseline | 0.760 | 0.747 | 0.708 |
| XLM-R zero-shot | 0.605 | 0.588 | 0.497 |
| XLM-R 500-shot (mean of 3 seeds) | 0.687 | 0.677 | 0.490 |
| Qwen2.5-1.5B ICL, 100-shot | 0.514 | 0.574 | 0.332 |
| Monolingual XLM-R upper bound | 0.750 | 0.769 | 0.490 |

Malayalam's flat, near-majority-class-floor performance across every few-shot and ICL condition — despite comparable or higher raw sample counts to Kannada and Tamil — is traced in the thesis to its severe class imbalance (3.94% offensive-class prevalence after label harmonisation, versus ~25% for the other two languages), not to insufficient data volume; see the LIME-based mechanistic analysis in the thesis for supporting evidence.

## Data sources & licensing

- **HASOC 2019/2020/2021**: released by the organisers of the Hate Speech and Offensive Content Identification shared task series (Mandl et al., 2019, 2020, 2021) for research use. See the [HASOC track pages](https://hasocfire.github.io/hasoc/official/index.html) for the original terms of use; redistribution here is for academic reproducibility purposes only.
- **DravidianCodeMix**: released by Chakravarthi et al. (2022), *"DravidianCodeMix: Sentiment analysis and offensive language identification dataset for Dravidian languages in code-mixed text,"* *Language Resources and Evaluation*, 56, 765–806. Original release: [github.com/bharathichezhiyan/DravidianCodeMix-Dataset](https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset).

If you reuse either dataset, please cite the original releases above, not this repository.

## Models used

- [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base) (Conneau et al., 2020)
- [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Qwen Team, 2024)

Both are loaded directly from the Hugging Face Hub; no modified weights are redistributed in this repository.

## Citation

If you use this code, please cite the thesis:

```
[Meet Goel]. ([2026]). Cross-Lingual and Few-Shot Offensive Language Detection in Low-Resource Dravidian Languages: An Empirical Study of XLM-R and Large Language Models. MSc Thesis, [Gisma University of Applied Sciences].
```

## Contact

[Meet Goel] — [meetgoel12345@gmail.com  / (https://www.gisma.com/de)]
