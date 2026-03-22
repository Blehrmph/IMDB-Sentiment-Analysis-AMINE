# IMDB Sentiment Analysis

Binary sentiment classification (positive / negative) on the IMDB 50k movie reviews dataset. The project is split across two homeworks, each building on the previous.

---

## Homework 1 — MLP with Lexicon Features

**Folder:** `MLP_folder/`

Reviews are not fed as raw text. Instead, each review is converted into **6 numeric features** using two lexicon-based tools:

| Feature | Tool |
|---|---|
| `vader_neg` | VADER |
| `vader_neu` | VADER |
| `vader_pos` | VADER |
| `vader_compound` | VADER |
| `textblob_polarity` | TextBlob |
| `textblob_subjectivity` | TextBlob |

These 6 features are standardized with `StandardScaler`, then fed into an **MLP classifier** (`hidden_layer_sizes=(64, 32), activation=relu`) from scikit-learn.

**Results saved to:** `MLP_folder/results/`

### How to run
```bash
pip install vaderSentiment textblob pandas numpy scikit-learn matplotlib
```
Open `MLP_folder/notebook.ipynb` and run all cells.

### Outputs
- `results/amine_assignment_model.pkl` — saved MLP + scaler
- `results/evaluation_results/confusion_matrix.png`
- `results/evaluation_results/classification_report.png`

---

## Homework 2 — Deep Learning Models on Raw Text

The following three models all work on **raw review text** rather than handcrafted features. Text is cleaned (HTML removed, lowercased, punctuation stripped), tokenized, and padded to a fixed length of 200 tokens.

Each model uses a **70 / 10 / 20** train / validation / test split and is trained with:
- `Adam` optimizer
- `EarlyStopping` (patience=3, restores best weights)
- `ReduceLROnPlateau` (factor=0.5, patience=2)

---

### Model 2 — LSTM

**Folder:** `model2_lstm/`

| Layer | Details |
|---|---|
| Embedding | 20 000 vocab, 128-dim, learned |
| Dropout | 0.2 |
| LSTM | 64 units |
| Dropout | 0.2 |
| Dense | 1 unit, sigmoid |

Captures **long-range sequential dependencies** in the review text.

**Results saved to:** `model2_lstm/results/`

### How to run
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```
Open `model2_lstm/notebook.ipynb` and run all cells.

---

### Model 3 — 1D CNN

**Folder:** `model3_cnn/`

| Layer | Details |
|---|---|
| Embedding | 20 000 vocab, 128-dim, learned |
| Conv1D | 128 filters, kernel 5, ReLU |
| BatchNormalization | — |
| Conv1D | 64 filters, kernel 3, ReLU |
| BatchNormalization | — |
| GlobalMaxPooling1D | — |
| Dropout | 0.3 |
| Dense | 64 units, ReLU |
| Dropout | 0.3 |
| Dense | 1 unit, sigmoid |

Detects **local n-gram patterns** across the sequence. Faster to train than LSTM.

**Results saved to:** `model3_cnn/results/`

### How to run
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```
Open `model3_cnn/notebook.ipynb` and run all cells.

---

### Model 4 — Bidirectional GRU + Word2Vec

**Folder:** `model4_gru_word2vec/`

| Layer | Details |
|---|---|
| Embedding | Pretrained Word2Vec (100-dim), frozen |
| Bidirectional GRU | 64 units, return sequences |
| Dropout | 0.3 |
| Bidirectional GRU | 32 units |
| Dropout | 0.3 |
| Dense | 64 units, ReLU |
| Dropout | 0.3 |
| Dense | 1 unit, sigmoid |

A **Word2Vec model** (vector_size=100, window=5) is trained on the training corpus using Gensim. Its vectors are loaded into the embedding layer as pretrained weights and frozen during training — the GRU then learns sentiment patterns on top of these semantic representations.

**Results saved to:** `model4_gru_word2vec/results/`

### How to run
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib gensim
```
Open `model4_gru_word2vec/notebook.ipynb` and run all cells.

---

## Project Structure

```
IMDB-Sentiment-Analysis/
│
├── MLP_folder/                  # Homework 1
│   ├── notebook.ipynb
│   ├── IMDB_Dataset.csv
│   └── results/
│       ├── amine_assignment_model.pkl
│       └── evaluation_results/
│
├── model2_lstm/                 # Homework 2 — LSTM
│   ├── notebook.ipynb
│   ├── README.md
│   └── results/
│
├── model3_cnn/                  # Homework 2 — 1D CNN
│   ├── notebook.ipynb
│   ├── README.md
│   └── results/
│
└── model4_gru_word2vec/         # Homework 2 — BiGRU + Word2Vec
    ├── notebook.ipynb
    ├── README.md
    └── results/
```

---

## Dataset

[IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) — 50 000 reviews, balanced (25k positive / 25k negative).
