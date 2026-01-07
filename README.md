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
- ✅ **Pruning mix**

### Stage 2 — Structured MLP Pruning
Notebook : `notebooks/03_mlp_pruning.ipynb`

Ce notebook applique le **pruning structuré des neurones MLP** sur deux types de modèles :

- **Modèle baseline + MLP pruning**
- **Modèle déjà head-pruned + MLP pruning**

Objectifs :
- Isoler l’impact du pruning MLP seul
- Évaluer l’effet cumulatif du pruning Attention + MLP
- Comparer les trade-offs qualité / compression entre les variantes

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

⚠️ **Important**  
Les modèles de base (checkpoint Hugging Face) **ne sont pas inclus dans le repository**.  
Ils sont automatiquement téléchargés dans les notebooks (ou doivent être fournis via Hugging Face) avant toute exécution.

### Prérequis
- Compte Kaggle avec GPU (T4 / P100 recommandé)
- Accès aux modèles Hugging Face (login HF si nécessaire)

### Ordre d’exécution recommandé

1. **Baseline & Importance Analysis**  
   `00-setup-01-baseline-and-importance.ipynb`  
   - Téléchargement du modèle de base  
   - Mesures baseline (qualité, latence, mémoire)  
   - Collecte des signaux d’importance

2. **Structured Head Pruning**  
   `02_head_pruning.ipynb`  
   - Pruning structuré des têtes d’attention (Q + O uniquement)  
   - Sauvegarde du modèle *head-pruned*

3. **Structured MLP Pruning**  
   `03_mlp_pruning.ipynb`  
   - Deux variantes de modèles sont générées :
     - **Baseline + MLP pruning**
     - **Head-pruned model + MLP pruning**
   - Comparaison directe entre :
     - pruning MLP seul
     - pruning Attention + MLP

4. **Recovery Tuning (LoRA léger)**  
   `04_recovery_lora.ipynb`  
   - Recovery LoRA sur un nombre limité de modèles
   - 1 epoch maximum

5. **Final Evaluation & Analysis**  
   `06_final_eval.ipynb`  
   - Comparaison finale :
     - baseline
     - head-pruned
     - MLP-pruned
     - mix pruning
     - après recovery LoRA


---

## Notes / Design Choices

- **Structured pruning** : réduction effective des matrices → impact réel sur taille/latence.
- **GQA-friendly attention pruning** : pruning sur **Q + O** uniquement, en gardant K/V cohérents.
- **Recovery LoRA** : correction légère, pas le cœur de la compression.

---

semantic_pruning_pipeline:
  pruning_targets:
    attention_heads:
      pruning_type: structured
      criterion: semantic_importance
      effect:
        - remove_entire_heads
        - reduce_attention_computation

    mlp_layers:
      pruning_type: structured
      criterion: intermediate_dimension_reduction
      effect:
        - reduce_hidden_dimension
        - remove_redundant_neurons

  recovery_stage:
    method: LoRA
    trainable_parameters: low_rank_adapters
    goal: recover_performance_after_pruning

  deployment:
    lora_merge: enabled
    inference_overhead: none


## License

MIT License
