# Model 3 — 1D Convolutional Neural Network

## Model Architecture

```
Input (token IDs, seq_len=200)
  └── Embedding(20000, 128)             ← learned from scratch
  └── Conv1D(128 filters, kernel=5, relu)
  └── BatchNormalization
  └── Conv1D(64 filters, kernel=3, relu)
  └── BatchNormalization
  └── GlobalMaxPooling1D               ← picks strongest feature per channel
  └── Dropout(0.3)
  └── Dense(64, relu)
  └── Dropout(0.3)
  └── Dense(1, sigmoid)                ← binary output
```

- **Two Conv1D layers**: extract local n-gram features (windows of 5 then 3 tokens). Stacking two allows the second layer to detect patterns of patterns.
- **BatchNormalization**: after each conv layer — normalizes activations, speeds up training, acts as a regularizer.
- **GlobalMaxPooling1D**: collapses the sequence dimension by taking the max value per filter — makes the model position-invariant (a sentiment phrase anywhere in the review is captured equally).
- **Dense head**: 64-unit ReLU with dropout before the sigmoid output.

## Techniques Applied

### Text Preprocessing
- HTML tag removal via regex
- Lowercasing
- Punctuation and digit removal (`[^a-z\s]`)
- Whitespace normalization

### Data Split
- Train: 70% (35,000) — Val: 10% (5,000) — Test: 20% (10,000)
- Explicit `validation_data` passed to `model.fit()`

### Training Optimizations
| Technique | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary cross-entropy |
| BatchNormalization | After each Conv1D |
| Dropout | 0.3 after pooling and dense |
| EarlyStopping | patience=3, monitors `val_loss`, restores best weights |
| ReduceLROnPlateau | factor=0.5, patience=2, min_lr=1e-6 |
| Batch size | 128 |
| Max epochs | 10 |

## Results
Saved to `results/model3_cnn/`:
- `confusion_matrix.png`
- `classification_report.png`
- `training_curves.png`
- `cnn_model.keras`
- `tokenizer.pkl`

## How to Run
1. Open `models/model3_cnn/notebook.ipynb`
2. Run all cells top to bottom
3. Dataset is read from `model/IMDB_Dataset.csv` (project root)
