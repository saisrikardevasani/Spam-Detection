# Email Spam Detection Using Machine Learning

## Project Overview

This project implements a machine learning-based email spam detection system using the Enron email corpus. The objective is to classify email messages as either **spam** (unsolicited/malicious) or **ham** (legitimate) through the application of multiple supervised learning algorithms, enabling comparative analysis to identify the optimal classification strategy.

### Key Objectives
- Develop a robust spam classification pipeline from raw email data to trained model predictions.
- Evaluate and compare six distinct machine learning algorithms for text classification performance.
- Provide a reproducible, well-documented codebase suitable for academic and corporate reference.

### Deliverables
| Deliverable | Description |
|---|---|
| `edt_project.ipynb` | Complete Jupyter notebook with end-to-end pipeline |
| `spam_ham_dataset.csv` | Pre-processed Enron email dataset (5,171 samples) |
| `Project_Report.md` | Formal project report with full analysis |
| `Spam_Detection_Literature_Review.pdf` | Academic literature review |
| `README.md` | This documentation file |

---

## Data Collection and Preparation

### Data Source
- **Dataset:** Enron Spam/Ham Email Corpus
- **Format:** CSV (`spam_ham_dataset.csv`)
- **Records:** 5,171 emails (3,672 ham / 1,499 spam)
- **Columns:** `label` (ham/spam), `text` (email content), `label_num` (0/1)

### Preprocessing Pipeline
1. **Text Cleaning:** Removal of all punctuation using `string.punctuation`.
2. **Stop Word Removal:** Filtering of English stop words via NLTK's stopwords corpus.
3. **Label Encoding:** Conversion of categorical labels (`ham`/`spam`) to binary integers (0/1).

### Feature Engineering
- **TF-IDF Vectorization:** Transformation of preprocessed text into a 50,342-dimensional TF-IDF feature matrix using `sklearn.feature_extraction.text.TfidfVectorizer`.
- **Word Cloud Visualization:** Generation of word frequency visualizations for both spam and ham classes to inform feature understanding.

---

## Model Development

Six supervised learning classifiers were implemented:

| Model | Algorithm | Library | Rationale |
|---|---|---|---|
| **SVC** | Support Vector Classifier (sigmoid kernel) | `sklearn.svm` | Effective for high-dimensional text data |
| **KNN** | K-Nearest Neighbors (k=49) | `sklearn.neighbors` | Non-parametric baseline classifier |
| **Naive Bayes** | Multinomial Naive Bayes (α=0.2) | `sklearn.naive_bayes` | Standard baseline for text classification |
| **Decision Tree** | CART (min_samples_split=7) | `sklearn.tree` | Interpretable, non-linear decision boundaries |
| **Logistic Regression** | L1-regularized (liblinear) | `sklearn.linear_model` | Sparse feature selection via L1 penalty |
| **Random Forest** | Ensemble of 31 trees | `sklearn.ensemble` | Variance reduction through bagging |

---

## Performance Summary

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|---|---|---|---|---|
| **Naive Bayes** | **98.58%** | 0.99 | 0.96 | 0.97 |
| SVC | 97.94% | 0.94 | 0.99 | 0.96 |
| Random Forest | 97.16% | 0.92 | 0.99 | 0.95 |
| KNN | 96.01% | 0.97 | 0.88 | 0.93 |
| Logistic Regression | 95.36% | 0.86 | 0.99 | 0.92 |
| Decision Tree | 94.46% | 0.87 | 0.94 | 0.91 |

**Best Model:** Multinomial Naive Bayes — highest overall accuracy (98.58%) with balanced precision and recall.

---

## Quick Start

```bash
# Clone the repository
git clone git@github.com:saisrikardevasani/Spam-Detection.git
cd Spam-Detection

# Install dependencies
pip install pandas numpy scikit-learn nltk wordcloud matplotlib jupyter

# Run the notebook
jupyter notebook edt_project.ipynb
```

---

## Project Structure
```
Spam-Detection/
├── README.md                              # Project documentation
├── Project_Report.md                      # Detailed project report
├── edt_project.ipynb                      # Main analysis notebook
├── spam_ham_dataset.csv                   # Enron email dataset
└── Spam_Detection_Literature_Review.pdf   # Academic literature review
```

---

## License
This project is developed for academic and research purposes.