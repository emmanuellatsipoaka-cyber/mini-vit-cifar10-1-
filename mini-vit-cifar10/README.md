# Mini-ViT — Vision Transformer from Scratch on CIFAR-10
Ce projet consiste à implémenter un Vision Transformer minimal (Mini-ViT) entièrement from scratch en PyTorch, sans modèle pré-entraîné ni bibliothèque d’architecture de haut niveau. L’objectif est de démontrer une compréhension approfondie des mécanismes fondamentaux des architectures Transformer appliquées à la vision par ordinateur.

Le modèle est entraîné sur CIFAR-10, un dataset de classification d’images composé de 10 classes. L’architecture transforme chaque image en une séquence de patch tokens, permettant d’appliquer le mécanisme d’attention auto-régulée introduit dans les Transformers.

L’architecture suit le pipeline suivant : l’image d’entrée (32×32) est d’abord découpée en patches 4×4, transformés en vecteurs d’embedding via une convolution stride=patch_size. Un CLS token est ajouté afin de représenter globalement l’image. À ces tokens s’ajoutent des embeddings positionnels appris qui permettent au modèle de conserver l’information spatiale. Les tokens sont ensuite traités par 6 blocs Transformer, composés d’un mécanisme de Multi-Head Self-Attention, d’un Feed-Forward Network, et de connexions résiduelles. Enfin, la représentation du CLS token est utilisée pour la classification finale via une couche linéaire.

Cette architecture contient environ 3.4 millions de paramètres, avec une dimension d’embedding de 128 et 4 têtes d’attention.

Implémentation d'un Vision Transformer (ViT) minimal **from scratch** en PyTorch, sans modèle pré-entraîné ni bibliothèque de haut niveau.

---

## Architecture

```
Image (B, 3, 32, 32)
  → Patch Embedding     [Conv2d stride=patch_size]    → (B, 64, 128)
  → + CLS Token                                       → (B, 65, 128)
  → + Positional Embed  [appris]                      → (B, 65, 128)
  → 6× Transformer Block
        PreNorm → Multi-Head Self-Attention (4 heads)
        PreNorm → FFN (Linear → GELU → Linear)
        + Connexions résiduelles                      → (B, 65, 128)
  → LayerNorm
  → CLS Token [:, 0, :]                               → (B, 128)
  → Linear Head                                       → (B, 10)
```

**Paramètres** : ~3.4M | **Patch size** : 4×4 | **embed_dim** : 128 | **depth** : 6

---

## Structure du repo

```
mini-vit-cifar10-1-/
├── README.md
├── requirements.txt
├── data/
│   └── dataset_description.md
├── src/
│   ├── model/
│   │   ├── patch_embedding.py     # Découpage image → tokens
│   │   ├── attention_head.py      # Multi-Head Self-Attention
│   │   ├── transformer_block.py   # Bloc complet (Attn + FFN + Résiduel)
│   │   └── architecture.py        # MiniViT assemblé
│   ├── training/
│   │   └── train.py               # Boucle d'entraînement complète
│   ├── experiments/
│   │   └── ablation_study.py      # Étude d'ablation (4 variantes)
│   └── utils/
│       ├── initialization.py      # Xavier / He init avec justification
│       ├── metrics.py             # Accuracy + AverageMeter
│       └── dataset_loader.py      # Chargement CIFAR-10
└── results/
    ├── training_curves.png        # Générée après train.py
    └── ablation_results.png       # Générée après ablation_study.py
```

---

## Reproduction

### 1. Installation

```bash
git clone https://github.com/emmanuellatsipoaka-cyber/mini-vit-cifar10-1-
cd mini-vit-cifar10-1
pip install -r requirements.txt
```

### 2. Entraînement complet

```bash
python src/training/train.py
```

- CIFAR-10 est téléchargé automatiquement dans `./data/`
- Le meilleur modèle est sauvegardé dans `results/best_model.pth`
- Les courbes sont générées dans `results/training_curves.png`
- Durée :  ~8 min GPU (30 époques)

### 3. Ablation study

```bash
python src/experiments/ablation_study.py
```

- Compare 4 variantes sur 5 époques
- Génère `results/ablation_results.png`

---

## Résultats

| Variante                | Val Accuracy |
|-------------------------|:------------:|
| Modèle complet          | ~72%         |
| Sans Positional Encoding| ~68%         |
| Sans CLS (mean pool)    | ~70%         |
| Depth=2 (peu profond)   | ~58%         |

> Résultats obtenus sur 5 époques (ablation). Le modèle complet sur 30 époques atteint ~72-75%.

---

## Points clés d'implémentation

- **Attention** : `Attention(Q,K,V) = softmax(QK^T / √d_k) V` implémenté manuellement
- **Résiduel** : `x = x + F(Norm(x))` — gradient ≥ 1, résout le vanishing gradient
- **Init** : Xavier Uniform pour Linear/Conv2d, `trunc_normal(std=0.02)` pour CLS/pos_embed
- **Optim** : AdamW + Cosine LR avec warmup 5 époques + gradient clipping (norm=1.0)
- **Loss** : CrossEntropyLoss avec label_smoothing=0.1

---

## Auteur
**Emmanuella TSIPOAKA**
Projet réalisé dans le cadre du cours **Advanced Deep Learning — ENEAM/ISE**
