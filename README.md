# Machine Learning Projects

A collection of applied machine-learning notebooks and a Streamlit prediction application.

## Projects

### 1. International Football Match & World Cup Prediction
**Notebook:** [WorldCUP_prediction_model.ipynb](WorldCUP_prediction_model.ipynb)

The notebook explores historical international football results and develops a match-outcome modelling workflow. It includes:
- Data cleaning and chronological ordering of match records.
- A custom Elo-rating calculation with competition-specific weighting.
- Pre-match feature engineering based on historical team performance.
- Logistic regression, random forest, and XGBoost model experiments.
- A time-based train/test split beginning in 2021.
- A Monte Carlo tournament simulation workflow.

**Reproducibility note:** The notebook expects a `results.csv` input file. Confirm the dataset version, date coverage, model metrics, and tournament assumptions by running all cells before relying on any prediction or performance claim.

### 2. Adult Income Classification
**Notebook:** [employee salary prediction.ipynb](employee%20salary%20prediction.ipynb)  
**Application:** [app.py](app.py)

Despite the notebook filename, the application interface and target classes show that this is an **income classification** project, predicting whether an individual's income falls at or below $50K or above $50K. It is not a continuous salary regression model.

The Streamlit app supports:
- Single-record predictions.
- Batch predictions from CSV or Excel uploads.
- Predicted income class and class probabilities.
- Downloading batch results as CSV.

Supporting files currently in the repository include `adult.csv`, `best_model.pkl`, and `requirements.txt`.

## Run the Streamlit application

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies.
4. Run the app.

```bash
git clone https://github.com/mdjees/MachineLearning.git
cd MachineLearning
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Technology
Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit, joblib, matplotlib, and seaborn.

## Responsible use and limitations
- The income-classification dataset contains demographic and socioeconomic attributes. Predictions may reflect historical dataset bias and should not be used for consequential decisions about individuals.
- The model file must be compatible with the installed scikit-learn version. Rebuild and version the model if loading fails.
- Do not upload personal or confidential data to the public application.
- Notebook outputs and reported metrics should be regenerated from the documented data and environment.

## Author
**Jeeshan Abrar**  
MBA, Finance & Analytics — Department of Business Economics, University of Delhi

- [GitHub](https://github.com/mdjees)
- [LinkedIn](https://www.linkedin.com/in/mdjeeshanabrar/)
