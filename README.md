# 📊 Statistical Presentation — ML Web App

An interactive machine learning web app built with **Streamlit** that lets you upload any CSV dataset, explore it visually, and train ML models — all from your browser, no code required.

---

## 🚀 Features

- **Auto Analysis** — query your dataset using natural language commands (powered by a semantic sentence similarity model)
- **Linear Regression** — fit and visualise a regression line with configurable learning rate and epochs
- **Polynomial Regression** — fit higher-degree curves to your data with adjustable polynomial degree
- **Logistic Regression** — classify categorical targets and view a confusion matrix with accuracy score
- **Decision Tree** — build and evaluate a decision tree classifier with tunable depth and split size
- **Column Dropping** — clean your dataset before modelling by dropping irrelevant features
- **Sample Datasets** — built-in links to Kaggle datasets so you can get started immediately

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [Pandas](https://pandas.pydata.org) | Data loading & manipulation |
| [NumPy](https://numpy.org) | Numerical computation |
| [Plotly](https://plotly.com/python) | Interactive charts & confusion matrices |
| [scikit-learn](https://scikit-learn.org) | ML models |
| [Sentence Transformers](https://www.sbert.net) | Natural language command parsing |

---

## ⚙️ Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Machine-Learning-Web-App.git
cd Machine-Learning-Web-App
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📖 How to Use

1. **Import Data** — upload a `.csv` file from your machine
2. **Select a purpose** — choose between Analysis, or one of the four ML models
3. **Tune hyperparameters** — adjust settings in the sidebar (learning rate, epochs, tree depth, etc.)
4. **Run** — hit the Run button and see your results, charts, and metrics instantly

### Natural Language Commands (Auto Analysis tab)
You can type plain English queries like:
- `"show me a scatter plot of age and salary"`
- `"what is the mean of score"`

The app uses a pre-trained semantic similarity model to interpret your intent.

---

## 📁 Project Structure

```
├── app.py               # Main Streamlit app & UI layout
├── functionality.py     # NLP command parsing, ML wrappers & helper functions
├── regressions.py       # Linear & polynomial regression implementations
├── classifications.py   # Logistic regression & decision tree implementations
├── visuals.py           # Plotly chart builders (regression line, confusion matrix)
├── requirements.txt     # Python dependencies
└── *.csv                # Sample datasets for testing
```

---

## 📸 Screenshots

> _Add screenshots here once deployed — a quick screen recording goes a long way!_

---

## 🙌 Acknowledgements

Sample datasets sourced from [Kaggle](https://www.kaggle.com). Semantic NLP powered by the `stsb-mpnet-base-v2` model via [Sentence Transformers](https://www.sbert.net).
