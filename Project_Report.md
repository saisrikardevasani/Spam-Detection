# Email Spam Detection: Comprehensive Project Report

**Author:** Sai Srikar Devasani  
**Date:** February 2026  
**Repository:** [github.com/saisrikardevasani/Spam-Detection](https://github.com/saisrikardevasani/Spam-Detection)

---

## 1. Project Overview

### 1.1 Objectives
The primary objective of this project is to develop, evaluate, and compare multiple machine learning classifiers for the automated detection of spam emails. The project leverages the Enron email corpus — a publicly available dataset comprising 5,171 labeled email messages — to train and benchmark six distinct classification algorithms.

### 1.2 Key Milestones
| Milestone | Description | Status |
|---|---|---|
| Data Acquisition | Sourcing and loading the Enron spam/ham dataset | ✅ Complete |
| Exploratory Data Analysis | Word cloud visualization and class distribution analysis | ✅ Complete |
| Text Preprocessing | Punctuation removal, stop word filtering, TF-IDF vectorization | ✅ Complete |
| Model Training | Training of six ML classifiers | ✅ Complete |
| Performance Evaluation | Accuracy, precision, recall, and F1-score computation | ✅ Complete |
| Documentation & Reporting | Comprehensive project documentation and literature review | ✅ Complete |

### 1.3 Deliverables
- Jupyter notebook (`edt_project.ipynb`) with the complete end-to-end pipeline.
- Preprocessed dataset (`spam_ham_dataset.csv`).
- Academic literature review (`Spam_Detection_Literature_Review.pdf`).
- Project report (this document) and README documentation.

---

## 2. Data Collection and Preparation

### 2.1 Data Source
The dataset originates from the **Enron email corpus**, one of the most widely used benchmarks in email spam classification research. The dataset is structured as a CSV file with the following schema:

| Column | Type | Description |
|---|---|---|
| `Unnamed: 0` | Integer | Original index identifier |
| `label` | String | Classification label (`ham` or `spam`) |
| `text` | String | Full email content including subject line |
| `label_num` | Integer | Binary encoding (0 = ham, 1 = spam) |

**Dataset Statistics:**
- Total samples: **5,171**
- Ham (legitimate) emails: **3,672** (71.0%)
- Spam emails: **1,499** (29.0%)
- Class imbalance ratio: ~2.45:1 (ham:spam)

### 2.2 Data Cleaning and Preprocessing

The following preprocessing pipeline was implemented to ensure data quality:

1. **Punctuation Removal:** All punctuation characters were stripped from the email text using Python's `string.punctuation` mapping via `str.maketrans()`.

2. **Stop Word Removal:** Common English stop words (e.g., "the", "is", "and") were removed using NLTK's curated stopwords corpus to reduce noise and dimensionality.

3. **Label Encoding:** Categorical string labels were converted to binary integer representation:
   - `ham` → `0`
   - `spam` → `1`

### 2.3 Feature Engineering

**TF-IDF Vectorization** was employed as the primary feature extraction technique:
- **Library:** `sklearn.feature_extraction.text.TfidfVectorizer`
- **Resulting feature space:** 50,342 unique terms
- **Method:** Term Frequency–Inverse Document Frequency weighting, which captures both local term importance (within a document) and global discriminative power (across the corpus).

**Word Cloud Visualization** was generated for both spam and ham classes to provide qualitative insight into the most frequent terms in each category. High-frequency spam terms included promotional and financial language, while ham terms reflected corporate communication patterns consistent with the Enron organization.

---

## 3. Model Development

### 3.1 Models Implemented

Six distinct supervised learning classifiers were selected to represent a diverse range of algorithmic approaches:

#### 3.1.1 Support Vector Classifier (SVC)
- **Algorithm:** SVM with sigmoid kernel function
- **Hyperparameters:** `kernel='sigmoid'`, `gamma=1.0`
- **Rationale:** SVMs are highly effective in high-dimensional feature spaces typical of text classification tasks. The sigmoid kernel provides non-linear decision boundaries.

#### 3.1.2 K-Nearest Neighbors (KNN)
- **Algorithm:** Instance-based learning with majority voting
- **Hyperparameters:** `n_neighbors=49`
- **Rationale:** Serves as a non-parametric baseline classifier. The large k-value (49) was selected to smooth the decision boundary and reduce sensitivity to noise in the high-dimensional TF-IDF space.

#### 3.1.3 Multinomial Naive Bayes (NB)
- **Algorithm:** Probabilistic classifier based on Bayes' theorem with multinomial likelihood
- **Hyperparameters:** `alpha=0.2` (Laplace smoothing)
- **Rationale:** Naive Bayes is the gold standard baseline for text classification due to its computational efficiency and strong empirical performance despite the conditional independence assumption.

#### 3.1.4 Decision Tree (DT)
- **Algorithm:** CART (Classification and Regression Trees)
- **Hyperparameters:** `min_samples_split=7`, `random_state=111`
- **Rationale:** Provides interpretable decision rules and can capture non-linear relationships without feature scaling.

#### 3.1.5 Logistic Regression (LR)
- **Algorithm:** Linear classifier with L1 regularization
- **Hyperparameters:** `solver='liblinear'`, `penalty='l1'`
- **Rationale:** L1 regularization induces sparsity in the coefficient vector, effectively performing feature selection — particularly valuable with the 50,342-dimensional feature space.

#### 3.1.6 Random Forest (RF)
- **Algorithm:** Bagging ensemble of decision trees
- **Hyperparameters:** `n_estimators=31`, `random_state=111`
- **Rationale:** Reduces variance compared to individual decision trees through bootstrap aggregation. Robust to overfitting in high-dimensional settings.

---

## 4. Model Training and Tuning

### 4.1 Training Configuration
- **Train/Test Split:** 85% training / 15% testing (`test_size=0.15`, `random_state=111`)
- **Training Set Size:** 4,395 samples
- **Test Set Size:** 776 samples
- **Feature Representation:** TF-IDF sparse matrix (50,342 features)

### 4.2 Evaluation Metrics
The following metrics were computed for each model:
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positives (spam) that are truly spam.
- **Recall (Sensitivity):** Proportion of actual spam correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.

### 4.3 Training Process
All six classifiers were trained using the same training data partition to ensure fair comparison. The `sklearn` library's `.fit()` method was used for training, and `.predict()` for inference on the held-out test set.

---

## 5. Performance Analysis

### 5.1 Detailed Results

#### Multinomial Naive Bayes — Best Performer (Accuracy: 98.58%)
|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ham (0) | 0.99 | 0.99 | 0.99 | 558 |
| Spam (1) | 0.99 | 0.96 | 0.97 | 218 |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **776** |

**Confusion Matrix:**
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| Actual Ham | 555 | 3 |
| Actual Spam | 8 | 210 |

#### Support Vector Classifier (Accuracy: 97.94%)
|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ham (0) | 1.00 | 0.97 | 0.99 | 558 |
| Spam (1) | 0.94 | 0.99 | 0.96 | 218 |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** | **776** |

**Confusion Matrix:**
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| Actual Ham | 544 | 14 |
| Actual Spam | 2 | 216 |

#### Random Forest (Accuracy: 97.16%)
|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ham (0) | 0.99 | 0.97 | 0.98 | 558 |
| Spam (1) | 0.92 | 0.99 | 0.95 | 218 |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** | **776** |

**Confusion Matrix:**
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| Actual Ham | 539 | 19 |
| Actual Spam | 3 | 215 |

#### K-Nearest Neighbors (Accuracy: 96.01%)
|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ham (0) | 0.96 | 0.99 | 0.97 | 558 |
| Spam (1) | 0.97 | 0.88 | 0.93 | 218 |
| **Weighted Avg** | **0.96** | **0.96** | **0.96** | **776** |

**Confusion Matrix:**
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| Actual Ham | 553 | 5 |
| Actual Spam | 26 | 192 |

#### Logistic Regression (Accuracy: 95.36%)
|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ham (0) | 1.00 | 0.94 | 0.97 | 558 |
| Spam (1) | 0.86 | 0.99 | 0.92 | 218 |
| **Weighted Avg** | **0.96** | **0.95** | **0.95** | **776** |

**Confusion Matrix:**
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| Actual Ham | 524 | 34 |
| Actual Spam | 2 | 216 |

#### Decision Tree (Accuracy: 94.46%)
|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ham (0) | 0.98 | 0.94 | 0.96 | 558 |
| Spam (1) | 0.87 | 0.94 | 0.91 | 218 |
| **Weighted Avg** | **0.95** | **0.94** | **0.95** | **776** |

**Confusion Matrix:**
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| Actual Ham | 527 | 31 |
| Actual Spam | 12 | 206 |

### 5.2 Comparative Analysis

| Rank | Model | Accuracy | F1 (Spam) | False Positives | False Negatives |
|---|---|---|---|---|---|
| 1 | **Naive Bayes** | **98.58%** | **0.97** | **3** | **8** |
| 2 | SVC | 97.94% | 0.96 | 14 | 2 |
| 3 | Random Forest | 97.16% | 0.95 | 19 | 3 |
| 4 | KNN | 96.01% | 0.93 | 5 | 26 |
| 5 | Logistic Regression | 95.36% | 0.92 | 34 | 2 |
| 6 | Decision Tree | 94.46% | 0.91 | 31 | 12 |

### 5.3 Key Observations and Trade-offs

1. **Naive Bayes achieves the highest overall accuracy (98.58%)** with the best balance of false positives and false negatives. Its computational simplicity and strong performance make it the recommended model for deployment.

2. **SVC and Random Forest exhibit high recall for spam (0.99)**, meaning they miss very few spam emails (only 2-3 false negatives). This is advantageous in environments where spam infiltration is costly.

3. **KNN shows the lowest spam recall (0.88)**, missing 26 spam emails out of 218. This suggests that distance-based methods may struggle in the extremely high-dimensional TF-IDF space.

4. **Logistic Regression produces the most false positives (34)**, which could result in legitimate emails being incorrectly flagged. However, its near-perfect spam recall (0.99) makes it suitable for high-security contexts where missing spam is more costly than occasional false alarms.

5. **The precision-recall trade-off** is most evident in SVC (high recall, moderate precision for spam) versus KNN (high precision, lower recall for spam).

---

## 6. Model Verification

### 6.1 Validation Methodology
- **Hold-out validation:** A stratified 85/15 train-test split was employed with `random_state=111` to ensure reproducibility and proportional class representation.
- **Test set size:** 776 samples (558 ham, 218 spam).

### 6.2 Verification Steps
1. All models were trained exclusively on the training partition (4,395 samples).
2. Predictions were generated on the unseen test partition (776 samples).
3. Confusion matrices were computed to identify specific error patterns.
4. Classification reports were generated with per-class precision, recall, and F1-score.

### 6.3 Issues Encountered and Resolutions
| Issue | Resolution |
|---|---|
| Hardcoded file path (`/content/...`) from Google Colab environment | Replaced with auto-download logic using `urllib.request` |
| Incorrect column rename (`v1`/`v2`) not matching dataset schema | Removed erroneous rename operation |
| Variable name bug in ham word cloud loop (`text.lower()` instead of `val.lower()`) | Corrected variable reference |

---

## 7. Documentation and Reporting

### 7.1 Documentation Practices
- **Code Comments:** In-line comments throughout the Jupyter notebook explain each processing step, parameter choice, and evaluation metric.
- **README.md:** Provides a concise project overview, quick-start instructions, performance summary, and repository structure.
- **Project Report (this document):** Formal, comprehensive analysis covering all phases of the ML pipeline.
- **Literature Review:** Academic paper (`Spam_Detection_Literature_Review.pdf`) reviewing prior work in spam detection methodologies.

### 7.2 Repository Structure
```
Spam-Detection/
├── README.md                              # Project overview and quick start
├── Project_Report.md                      # This comprehensive report
├── edt_project.ipynb                      # Jupyter notebook (executed with outputs)
├── spam_ham_dataset.csv                   # Enron email dataset (5,171 samples)
└── Spam_Detection_Literature_Review.pdf   # Academic literature review
```

---

## 8. Conclusion

### 8.1 Project Outcomes
This project successfully demonstrates the end-to-end development and evaluation of an email spam detection system. Key outcomes include:

- **Multinomial Naive Bayes** emerged as the top-performing classifier with **98.58% accuracy** and a **0.97 F1-score** for spam detection, confirming its well-established effectiveness for text classification tasks.
- All six models achieved accuracy above **94%**, indicating that TF-IDF feature representation effectively captures discriminative patterns between spam and legitimate email content.
- The project codebase is fully reproducible, with automated dataset download and clearly documented preprocessing steps.

### 8.2 Lessons Learned
1. **Feature engineering is paramount:** TF-IDF vectorization yielded a 50,342-dimensional feature space that proved highly effective, even with relatively simple classifiers.
2. **Probabilistic models excel at text classification:** Naive Bayes, despite its independence assumption, consistently outperforms more complex models on text data — a finding corroborated by the broader NLP literature.
3. **Data quality directly impacts model reliability:** The three bugs identified during code review (incorrect file paths, erroneous column operations, and variable name errors) would have silently corrupted results if undetected.

### 8.3 Recommendations for Future Work
- **Deep Learning Approaches:** Evaluate transformer-based models (e.g., BERT, RoBERTa) for contextual email representation.
- **Cross-Validation:** Implement k-fold stratified cross-validation for more statistically robust performance estimates.
- **Feature Enrichment:** Incorporate metadata features (sender reputation, email headers, attachment types) alongside textual content.
- **Real-Time Deployment:** Develop a REST API or email server integration for live spam filtering.
- **Class Imbalance Handling:** Apply SMOTE or class-weight adjustments to address the 2.45:1 ham-to-spam ratio.

### 8.4 Alignment with Organizational Goals
This project directly supports organizational objectives in:
- **Cybersecurity:** Automated spam detection reduces exposure to phishing attacks and social engineering threats.
- **Operational Efficiency:** Filtering spam emails reduces inbox clutter, improving employee productivity.
- **Compliance:** Accurate email classification supports regulatory requirements for data governance and information security.
- **Research & Innovation:** The comparative analysis framework provides a reusable methodology for evaluating ML models across other classification domains.

---

*End of Report*
