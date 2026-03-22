# Model 4 — Bidirectional GRU with Pretrained Word2Vec Embeddings

## Model Architecture

```
Input (token IDs, seq_len=200)
  └── Embedding(20001, 100)  ← initialized from Word2Vec, frozen
  └── Bidirectional(GRU(64, return_sequences=True))
  └── Dropout(0.3)
  └── Bidirectional(GRU(32))
  └── Dropout(0.3)
  └── Dense(64, relu)
  └── Dropout(0.3)
  └── Dense(1, sigmoid)      ← binary output
```

- **Pretrained Embedding**: Word2Vec vectors (100-dim) trained on the training corpus using Gensim. Weights are loaded into the Keras Embedding layer and frozen (`trainable=False`) so the semantic structure is preserved.
- **Bidirectional GRU**: reads the sequence both left-to-right and right-to-left, concatenating both hidden states — captures context from both directions.
- **Stacked BiGRU**: first layer returns full sequences so the second layer can process them; second layer returns a single summary vector.
- **GRU vs LSTM**: GRU has fewer parameters (no separate cell state gate), trains faster while still capturing long-range dependencies.
- **Dense head**: 64-unit ReLU with dropout before the sigmoid output.

## Techniques Applied

### Text Preprocessing
- HTML tag removal via regex
- Lowercasing
- Punctuation and digit removal (`[^a-z\s]`)
- Whitespace normalization

### Pretrained Word2Vec (Gensim)
- Trained on the 35,000 training reviews only (no data leakage)
- `vector_size=100`, `window=5`, `min_count=2`, `epochs=10`
- Embedding matrix built by looking up each Keras tokenizer word index in the Word2Vec vocabulary
- Words not found in Word2Vec remain as zero vectors

### Data Split
- Train: 70% (35,000) — Val: 10% (5,000) — Test: 20% (10,000)
- Explicit `validation_data` passed to `model.fit()`

### Training Optimizations
| Technique | Value |
|---|---|
| Pretrained embeddings | Word2Vec via Gensim, frozen |
| Bidirectional GRU | Reads sequence in both directions |
| Stacked BiGRU | 2 layers for deeper sequential representation |
| Dropout | 0.3 after each GRU and dense layer |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary cross-entropy |
| EarlyStopping | patience=3, monitors `val_loss`, restores best weights |
| ReduceLROnPlateau | factor=0.5, patience=2, min_lr=1e-6 |
| Batch size | 128 |
| Max epochs | 10 |

## Results
Saved to `results/model4_gru_word2vec/`:
- `confusion_matrix.png`
- `classification_report.png`
- `training_curves.png`
- `gru_word2vec_model.keras`
- `tokenizer.pkl`
- `word2vec.model`

## How to Run
1. Open `models/model4_gru_word2vec/notebook.ipynb`
2. Run all cells top to bottom
3. Dataset is read from `model/IMDB_Dataset.csv` (project root)
4. Word2Vec trains automatically (~1-2 min) before the Keras model
