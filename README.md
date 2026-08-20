# IMDB Sentiment Classification

A sentiment classification experiment on the IMDB movie review dataset.

This project compares a traditional machine learning baseline using
TF-IDF + Logistic Regression with a fine-tuned DistilBERT transformer model.

## Objective

The goal is to compare a classical NLP approach with a pretrained
Transformer-based approach for binary sentiment classification.

The models classify movie reviews as either:

- Positive
- Negative

## Dataset

The project uses the [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
available through Hugging Face Datasets.

The original training split is divided into:

- 80% training
- 20% validation

The original test split is kept separate for final evaluation.

## Approach

### 1. TF-IDF + Logistic Regression

The baseline model uses:

- TF-IDF vectorization
- Maximum 5,000 features
- Logistic Regression

Several values of the regularization parameter `C` are evaluated
on the validation set, and the best-performing value is selected.

The final baseline is then evaluated on the test set.

### 2. DistilBERT

The second approach fine-tunes:

`distilbert-base-uncased`

The pipeline consists of:

- DistilBERT tokenizer
- Dynamic padding using `DataCollatorWithPadding`
- PyTorch `DataLoader`
- AdamW optimizer
- Learning rate: `2e-5`
- Batch size: `8`
- GPU training when CUDA is available

The model is trained for multiple epochs and evaluated on the validation
and test sets.

## Results

| Model | Test Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | 88.16% |
| DistilBERT | 91.88% |

DistilBERT achieved higher test accuracy than the TF-IDF + Logistic
Regression baseline in this experiment.

Additional evaluation metrics for DistilBERT include:

- Accuracy
- Precision
- Recall
- F1 Score

## Key Takeaways

- TF-IDF + Logistic Regression provides a strong and computationally
  inexpensive baseline for sentiment classification.
- DistilBERT performs better on this dataset in the experiment.
- Pretrained Transformer models can capture contextual information that
  traditional bag-of-words approaches do not.
- The trade-off is increased computational cost and training complexity.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
