# Model 2 — LSTM with Learned Embeddings

## Model Architecture

```
Input (token IDs, seq_len=200)
  └── Embedding(20000, 128)         ← learned from scratch
  └── Dropout(0.2)
  └── LSTM(64)
  └── Dropout(0.2)
  └── Dense(1, sigmoid)             ← binary output
```

- **Embedding**: maps each word index to a 128-dim vector, learned during training.
- **LSTM(64)**: captures long-range sequential dependencies in the review text.
- **Dropout(0.2)**: applied after embedding and after LSTM to reduce overfitting.
- **Dense(sigmoid)**: outputs a probability for positive/negative sentiment.

## Techniques Applied

### Text Preprocessing
- HTML tag removal (`<br />` etc.) via regex
- Lowercasing and whitespace normalization
- Keras `Tokenizer` — top 20,000 words, unknown words mapped to `<OOV>`
- Sequences padded/truncated to length 200

### Training Optimizations
| Technique | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary cross-entropy |
| EarlyStopping | patience=3, monitors `val_loss`, restores best weights |
| ReduceLROnPlateau | factor=0.5, patience=2, min_lr=1e-6 |
| Batch size | 128 |
| Max epochs | 10 |

## Results
Saved to `results/model2_lstm/`:
- `confusion_matrix.png`
- `classification_report.png`
- `training_curves.png`
- `lstm_model.keras`
- `tokenizer.pkl`

## How to Run
1. Open `models/model2_lstm/notebook.ipynb`
2. Run all cells top to bottom
3. Dataset is read from `model/IMDB_Dataset.csv` (project root)
