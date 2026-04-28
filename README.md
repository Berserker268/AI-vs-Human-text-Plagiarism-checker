# 🤖 AI vs Human Text Detector

A machine learning project that classifies text as either **AI-generated** or **Human-written** using stylometric and linguistic features — no vectorization, no embeddings, just writing patterns.

---

## 📌 Overview

This project explores whether AI-generated and human-written essays can be distinguished purely through **structural and stylistic features** of text. Instead of bag-of-words or transformer-based approaches, we extract numerical metrics that capture writing behavior, and train multiple classical ML models on them.

---

## 📂 Dataset

- **Source:** `AI_Human.csv`
- **Text column:** Essay text (human or AI generated)
- **Size**: 487,235 essay samples
- **Class distribution**: 62.7% Human / 37.3% AI
- **Target column:** `generated`
  - `0` = Human Written
  - `1` = AI Generated

Download the dataset from Kaggle:
- [Ai vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data)
---
## 🔍 Features Engineered

Using **spaCy**, **textstat**, and **NLTK**, the following stylometric features are extracted from each essay:

| Feature | Description |
|---|---|
| `punctuation density` | Ratio of punctuation marks to total words |
| `stopword` | Ratio of stopwords to total words |
| `noun density` | Ratio of nouns to total words |
| `verb density` | Ratio of verbs to total words |
| `adj density` | Ratio of adjectives to total words |
| `sentence length` | Average number of words per sentence |
| `ttr` | Type-Token Ratio — measures lexical diversity |
| `fk_grade` | Flesch-Kincaid readability grade |
| `fog_index` | Gunning Fog Index — penalizes complex words |
| `burstiness` | Standard deviation of sentence lengths |
| `entropy` | Unpredictability of word choices |
| `avg_word_len` | Average number of characters per word |

Key insight: AI tends to produce **consistent, predictable sentence lengths** (low burstiness), while humans write with more variability.

---

## 🧪 Models Trained

Six classifiers were trained and evaluated:

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 87% | 0.86 |
| SVM (LinearSVC) | 87% | 0.85 |
| Gradient Boosting | 91% | 0.90 |
| Voting Ensemble | 93% | 0.92 |
| XGBoost | 94% | 0.94 |
| **Random Forest** | **98%** | **0.98** |

Models were also retrained with **regularization** (reduced depth, subsampling, L1/L2 penalties) and on a **reduced feature set** to combat overfitting.

---

## 📊 Evaluation

Each model was evaluated with:
- Classification report (precision, recall, F1)
- Confusion matrix
- Train vs. test accuracy gap (overfitting check)
- Feature importance plots

An 80/20 stratified train-test split was used with `StandardScaler` normalization applied to all features.

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/ai-text-detector.git
cd ai-text-detector
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Requirements

```
pandas
numpy
scikit-learn
xgboost
spacy
textstat
nltk
seaborn
matplotlib
tqdm
```

---

## 🚀 Usage

1. Place your dataset (`AI_Human.csv`) in the project directory.
2. Open and run `AI_Text_Detector.ipynb` end to end.
3. The notebook will:
   - Perform EDA and text length analysis
   - Extract stylometric features
   - Train and evaluate all models
   - Visualize feature importance

---

## 📁 Project Structure

```
ai-text-detector/
├── AI_Text_Detector.ipynb   # Main notebook
├── Final Report.pdf   # week 4 report
├── README.md
├── Report.pdf      # week 3 report
├── Research Work.pdf     # week 2 report
└── requirements.txt   # required libraries
```

---

## 🔬 Key Findings

- **Random Forest** is the best performer with 98% accuracy and only 442 false positives out of 97,447 test samples
- **Burstiness** and **entropy** are the most theoretically grounded features — humans vary sentence length more and use less predictable vocabulary
- **avg_word_len** emerged as the strongest single predictor in tree-based models
- **Feature reduction** (dropping verb_density, ttr, noun_density, stopword) caused a 1% drop — all 12 features contribute complementary information
- **No significant overfitting** detected — all models showed less than 2% gap between train and test accuracy
- **Regularization** on Gradient Boosting and XGBoost successfully reduced avg_word_len dominance and distributed importance more evenly across features
---

## 📚 References

- Bennet, M. (2025). Feature-Based Detection of AI-Generated Text. ResearchGate.
- Masrour, E., Emi, B., & Spero, M. (2025). DAMAGE: Detecting Adversarially Modified AI Generated Text. GenAIDetect.
- Wu, J. (2025). A Survey on LLM-Generated Text Detection. ACL Anthology.
- Kaggle Dataset: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

---

## 🙋 Author

**Berserker268**
Made as part of CSE427 Lab Project.

Feel free to fork, star ⭐, and contribute!

