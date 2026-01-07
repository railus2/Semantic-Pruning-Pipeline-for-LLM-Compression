# Semantic Structured Pruning Pipeline for LLM Compression

Un pipeline complet de **compression de LLM** basé sur du **structured semantic pruning** (pruning de têtes d’attention et neurones MLP), avec **recovery tuning léger (LoRA)** et **évaluation qualité ↔ performance**.

> Objectif : compresser “réellement” le modèle (paramètres / FLOPs / latence), pas juste masquer des poids.

---

## Overview

Contrairement aux pipelines classiques (quantization / distillation), ce projet **modifie l’architecture** en supprimant des composants du réseau en fonction de leur **utilité sémantique**.

Pipeline (vue haut niveau) :

Baseline LLM
│
▼ Importance Analysis (semantic-based)
Ranking heads / neurons
│
▼ Structured Pruning
Smaller LLM (real reduction)
│
▼ Recovery (LoRA, light)
Recovered performance
│
▼ Final Evaluation
Quality vs Compression vs Latency

markdown
Copier le code

---

## Pipeline Stages (notebooks)

### Stage 0 — Baseline & Importance Analysis
Notebook : `notebooks/00-setup-01-baseline-and-importance.ipynb`

- Setup Kaggle (GPU)
- Baseline metrics (ex: perplexity / latence / mémoire)
- Collecte de signaux d’importance (activations, stats, etc.) pour préparer le ranking

### Stage 1 — Structured Head Pruning (Attention)
Notebook : `notebooks/02_head_pruning.ipynb`

Ce notebook fait **du structured head pruning** avec les contraintes suivantes :

- ✅ **Pruning structuré**
- ✅ **Q + O uniquement** (compatible **GQA**)
- ✅ taux principal : **-20% query heads**
  - exemple : **16 → 12 query heads**
  - **K/V restent à 2 heads** (GQA)
- ❌ pas de mix attention+MLP ici
- ❌ pas de -40% dans la version actuelle

### Stage 2 — Structured MLP Pruning
Notebook : `notebooks/03_mlp_pruning.ipynb`

- Pruning structuré côté MLP
- Expériences typiques : **-10% / -20%**
- Comparaison directe vs head pruning (-20%)

### Stage 3 — Recovery Tuning (LoRA léger)
Notebook : `notebooks/04_recovery_lora.ipynb`

- LoRA léger, peu d’époques
- Le pruning reste la compression principale ; LoRA sert à **récupérer** la performance perdue

### Stage 4 — Final Evaluation
Notebook : `notebooks/06_final_eval.ipynb`

- Évaluation finale multi-axes :
  - Qualité (perplexity / comportement instruction-following selon setup)
  - Coût inference (latence, mémoire)
  - Taille (params, fichiers)
- Analyse des trade-offs et ablations

---

## Repository Structure

semantic-llm-pruning/
├── notebooks/
│ ├── 00-setup-01-baseline-and-importance.ipynb
│ ├── 02_head_pruning.ipynb
│ ├── 03_mlp_pruning.ipynb
│ ├── 04_recovery_lora.ipynb
│ └── 06_final_eval.ipynb
│
├── models/
│ └── baseline/
│ └── .gitkeep
│
├── results/
│ └── ablation_studies/
│ ├── baseline.csv
│ ├── head_pruning.csv
│ ├── mlp_pruning.csv
│ ├── final_eval.csv
│ ├── perplexity_baseline.json
│ └── latency_baseline.json
│
└── README.md

yaml
Copier le code

---

## Results

Les fichiers de résultats et ablations sont disponibles ici :

- `results/ablation_studies/baseline.csv`
- `results/ablation_studies/head_pruning.csv`
- `results/ablation_studies/mlp_pruning.csv`
- `results/ablation_studies/final_eval.csv`
- `results/ablation_studies/perplexity_baseline.json`
- `results/ablation_studies/latency_baseline.json`

> Les métriques exactes dépendent du modèle/dataset utilisés dans les notebooks Kaggle.

---

## How to Run (Kaggle)

1. Ouvrir les notebooks dans Kaggle
2. Exécuter dans l’ordre :
   1) `00-setup-01-baseline-and-importance.ipynb`  
   2) `02_head_pruning.ipynb`  
   3) `03_mlp_pruning.ipynb`  
   4) `04_recovery_lora.ipynb`  
   5) `06_final_eval.ipynb`

---

## Notes / Design Choices

- **Structured pruning** : réduction effective des matrices → impact réel sur taille/latence.
- **GQA-friendly attention pruning** : pruning sur **Q + O** uniquement, en gardant K/V cohérents.
- **Recovery LoRA** : correction légère, pas le cœur de la compression.

---

## License

MIT License
