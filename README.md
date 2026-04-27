# AI-vs-Human-text-Plagiarism-checker

## Dataset

Download the dataset from Kaggle:
- [Ai vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data)

### Quick Start
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d [dataset-id]
unzip [dataset-name].zip
```
# 🤖 AI vs Human Text Detector

A machine learning project that classifies text as either **AI-generated** or **Human-written** using stylometric and linguistic features — no vectorization, no embeddings, just writing patterns.

---

## 📌 Overview

This project explores whether AI-generated and human-written essays can be distinguished purely through **structural and stylistic features** of text. Instead of bag-of-words or transformer-based approaches, we extract numerical metrics that capture writing behavior, and train multiple classical ML models on them.

---

## 📂 Dataset

- **Source:** `AI_Human.csv`
- **Text column:** Essay text (human or AI generated)
- **Target column:** `generated`
  - `0` = Human Written
  - `1` = AI Generated

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

- Logistic Regression
- Random Forest
- Gradient Boosting
- Linear SVM
- XGBoost
- Voting Ensemble (all of the above)

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
├── AI_Human.csv             # Dataset (not included)
├── new_df_features.csv      # Exported feature dataset (generated)
└── README.md
```

---

## 💡 Key Findings

- **Burstiness** and **entropy** are among the strongest discriminating features — humans write with more variable sentence lengths and less predictable word choices.
- **Type-Token Ratio (TTR)** reveals AI's lower lexical diversity.
- The **Ensemble model** combining all classifiers achieved the best overall performance.
- Regularized Gradient Boosting and XGBoost reduced overfitting while maintaining competitive accuracy.

---


