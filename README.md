# ⚽ Football Player Wages Prediction & EDA

An end-to-end data science and machine learning project that analyzes football player salaries, extracts key exploratory insights, and predicts wages using optimized regression algorithms.

---

## 📌 Project Overview

This project utilizes a structured dataset containing professional football player profiles, including their clubs, nations, positions, appearances, caps, and current wages. The workflow encompasses:

1. **Robust Data Preprocessing & Cleaning:** Handling duplicates, formatting structural anomalies (e.g., converting text-based currency wages to numeric values), and handling categorical encoding.
2. **Exploratory Data Analysis (EDA):** In-depth distribution checks (age, position-based wages, club demographics) coupled with rich matplotlib visualisations.
3. **Machine Learning Pipeline:** Splitting, feature scaling, and regression modeling using SVR (Support Vector Regression) and Random Forest Regressor.
4. **Hyperparameter Tuning:** Utilizing `GridSearchCV` to locate optimal estimators for both models and assessing performance using MAE and MSE metrics.

---

## 📊 Dataset Features

The project is built on `SalaryPrediction.csv`, which holds the following features:

- `Wage`: Weekly/annual salary of the football player (target variable).
- `Age`: Player age.
- `Club`: Associated club (e.g., PSG, Real Madrid, Vigo).
- `League`: Major leagues (e.g., Ligue 1, La Liga, Premier League).
- `Nation`: Nationality.
- `Position`: Player's tactical role (Forward, Midfielder, Defender, Goalkeeper).
- `Apps`: Total club appearances.
- `Caps`: International caps (national team matches).

---

## 🛠️ Tech Stack & Dependencies

The project uses the modern Python packaging tool `uv` and is configured with standard data science libraries:

- **Python**: `>=3.13`
- **Pandas**: Data manipulation and clean-up.
- **NumPy**: Mathematical operations.
- **Matplotlib**: Rich data plotting.
- **Scikit-Learn**: preprocessing (`LabelEncoder`, `StandardScaler`), model selection (`train_test_split`, `GridSearchCV`), models (`SVR`, `RandomForestRegressor`), and evaluation metrics.

---

## 📂 Project Structure

```bash
Football-Player-Wages/
│
├── SalaryPrediction.csv     # Raw tabular football player dataset
├── project.ipynb            # Core Jupyter Notebook containing EDA & ML models
├── main.py                  # Standard entrypoint placeholder
├── pyproject.toml           # Project metadata & dependency declarations
├── uv.lock                  # Lockfile mapping exact dependency versions
└── README.md                # Comprehensive project documentation
```

---

## 🔍 Key Insights from EDA

- **Wage Cleaning:** Raw values like `46,427,000` were processed, stripping string punctuation to cast values into standard integers.
- **Demographics:** Detailed age histograms show the primary distribution of active professional footballers.
- **Club Age Profiling:** Top clubs with the highest average player ages are graphed and analyzed.
- **Wage Distribution by Position:** Midfielders and forwards rank among the highest average earners across the dataset.

---

## 🤖 Modeling & Evaluation

The predictive performance of two core algorithms was evaluated:

1. **Support Vector Regressor (SVR):** Tuned via `GridSearchCV` on kernels (`linear`, `rbf`, `poly`).
2. **Random Forest Regressor:** Built to leverage ensemble decision trees, optimized across multiple `n_estimators` (estimators tested: `[2, 5, 7, 10]`) and depth levels.

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have Python `>=3.13` and optionally `uv` (recommended) or `pip` installed.

### 2. Installation
Clone this repository and sync the virtual environment:
```bash
git clone https://github.com/MustafaKocamann/Football-Player-Wages.git
cd Football-Player-Wages
```

If using `uv`:
```bash
uv sync
```

Alternatively, install dependencies manually:
```bash
pip install pandas numpy matplotlib scikit-learn ipykernel
```

### 3. Running the Project
Open the Jupyter notebook environment and run the cells within `project.ipynb`:
```bash
jupyter notebook project.ipynb
```
