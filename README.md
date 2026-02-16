# IMDB Sentiment Analysis (Lexicon Features + Logistic Regression)

This project builds a sentiment classifier for IMDB reviews using VADER and TextBlob scores as numeric features, then trains a logistic regression model from scratch in a Jupyter notebook.

## Overview
- Text is converted to 6 numeric features using VADER and TextBlob.
- A logistic regression model is trained with forward propagation, binary cross-entropy loss, and gradient descent.
- Evaluation outputs are saved as images.
- Trained parameters are saved to disk.

## Project Structure
- `model/notebook.ipynb`: main notebook (feature extraction, training, evaluation, saving).
- `model/IMDB_Dataset.csv`: dataset file.
- `results/amine_assignment_model.pkl`: saved model parameters and scaler.
- `results/evaluation_results/confusion_matrix.png`: evaluation image.
- `results/evaluation_results/classification_report.png`: evaluation image.

## Requirements
Install dependencies:
```bash
pip install vaderSentiment textblob pandas numpy scikit-learn matplotlib
```

## How To Run
1. Open `model/notebook.ipynb`.
2. Run all cells from top to bottom.
3. The model and evaluation images will be saved under `results/` in the project root.

## Outputs
- Model file: `results/amine_assignment_model.pkl`
- Evaluation images:
  - `results/evaluation_results/confusion_matrix.png`
  - `results/evaluation_results/classification_report.png`

## Notes
- The notebook expects to be run from within the `model/` folder.
- The save logic automatically writes outputs to the project root `results/` folder.
