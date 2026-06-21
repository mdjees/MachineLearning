# World Cup 2026 Prediction Engine

A machine learning pipeline that predicts the FIFA World Cup 2026 winner by simulating the full 48-team tournament 10,000 times. Built from scratch in Python using a custom Elo rating system, feature engineering on 150 years of match data, and an XGBoost classifier.

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | 48,000+ international matches (1872–2026) |
| Model | XGBoost (test accuracy: ~56%) |
| Simulations | 10,000 full tournament runs |
| Teams | 48 (2026 format) |

> Charts for win probabilities, group exit risk, Elo rankings, and model comparison are saved automatically when you run the notebook.

---

## How It Works

### 1. Custom Elo Rating System

Every team starts at 1500. Ratings update after every match in history using the standard Elo formula, with K-factors tuned by tournament importance:

```
Expected_A  =  1 / (1 + 10 ^ ((Rating_B - Rating_A) / 400))
New_Rating  =  Old_Rating + K × (Actual_Score - Expected_Score)
```

| Tournament | K-factor |
|-----------|----------|
| FIFA World Cup | 60 |
| Euros / Copa América | 55 |
| WC Qualification | 45 |
| Nations League | 30 |
| Friendly | 20 |

Home teams receive a +50 Elo boost on non-neutral grounds.

---

### 2. Feature Engineering

16 features are computed per match using only historical data before that match — no data leakage:

| Feature | Description |
|---------|-------------|
| `elo_diff` | Elo rating gap between teams |
| `home_elo` / `away_elo` | Absolute Elo ratings |
| `home_win_rate` / `away_win_rate` | Win % in last 15 games |
| `win_rate_diff` | Difference in win rates |
| `home_form` / `away_form` | Points in last 5 games (W=1, D=0.5, L=0) |
| `form_diff` | Difference in form scores |
| `home_goals_scored` / `away_goals_scored` | Avg goals scored, last 10 games |
| `home_goals_conceded` / `away_goals_conceded` | Avg goals conceded, last 10 games |
| `home_attack_vs_def` | Home attack minus away defensive record |
| `away_attack_vs_def` | Away attack minus home defensive record |
| `is_neutral` | 1 if neutral venue, 0 if home team present |

---

### 3. Machine Learning Models

Three classifiers trained on the same feature set, with a **time-based train/test split** (train on matches before 2021, test on 2021+):

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| Logistic Regression | ~52% | Ignored draws almost entirely |
| Random Forest | ~54% | 300 trees, `max_depth=12` |
| **XGBoost** | **~56%** | **Selected for simulation** |

> Baseline (random guess in a 3-class problem) = 33%. Hitting 56% on football data is strong.

XGBoost was selected for the simulation because it produces the best-calibrated probabilities for win / draw / loss — which directly feed the Monte Carlo engine.

---

### 4. Monte Carlo Tournament Simulation

The full 2026 World Cup is simulated 10,000 times:

**Group Stage**
- Round-robin within each of the 12 groups
- Match outcomes drawn probabilistically from XGBoost predictions
- Scorelines generated via Poisson distribution, adjusted to match result
- Top 2 from each group (24 teams) + best 8 third-place teams = 32 advance

**Knockout Stage (Round of 32 → Final)**
- Must produce a winner each match
- Draws go to a simulated penalty shootout (50/50 with a slight Elo-based edge)

**Win probability** = how often each team won across all 10,000 runs.

---

## Project Structure

```
wc2026-prediction-model/
│
├── wc2026_prediction_model.py   # Main script — run top to bottom in Colab
├── README.md
│
└── outputs/                     # Auto-generated when you run the script
    ├── elo_ratings.png
    ├── model_comparison.png
    ├── win_probabilities.png
    ├── group_exit_risk.png
    └── probability_heatmap.png
```

---

## Getting Started

### Platform

Google Colab is recommended. No local setup required.

### Data

Download `results.csv` from Kaggle:

```
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
```

Upload it to your Colab session when prompted.

### Run

```bash
# Install dependencies (run once)
pip install xgboost seaborn plotly

# Then open wc2026_prediction_model.py in Colab
# and run each cell top to bottom
```

> Important: cells must be run in order. Each cell depends on variables created by the one before it. Use Runtime → Restart and run all to be safe.

### Update the Groups

Edit `WC2026_GROUPS` in the simulation cell to match the actual draw. Team names must match the dataset exactly. Run this to check:

```python
print(sorted(df['home_team'].unique()))
```

Common mismatches: `Czech Republic` (not Czechia), `Turkey` (not Türkiye), `Bosnia-Herzegovina`, `Curacao`.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas + numpy | Data wrangling, Elo computation |
| scikit-learn | Logistic Regression, Random Forest, scaling |
| XGBoost | Primary prediction model |
| matplotlib + seaborn | Visualisations |
| Google Colab | Execution environment |

---

## Key Design Decisions

**Why a custom Elo system instead of using existing ratings?**
Full control over tournament weights. Existing ratings (e.g. World Football Elo) treat all competitions more uniformly. The custom K-factor system lets the model correctly value a World Cup win as three times more significant than a friendly.

**Why time-based train/test split?**
A random split would allow the model to see future matches during training — effectively cheating. Splitting at 2021 ensures the test set is entirely unseen data, which is how the model would behave in real deployment.

**Why XGBoost over Random Forest?**
Both produce similar accuracy, but XGBoost's probability estimates (via `predict_proba`) are better calibrated. Since the simulation relies on drawing outcomes from predicted probabilities, calibration matters more than raw accuracy.

**Why 10,000 simulations?**
Enough runs for the win probability estimates to stabilise. At 1,000 runs, results fluctuate by ±2–3%. At 10,000, they stabilise to within ±0.5%.

---

## Limitations

- Model accuracy is bounded by the inherent unpredictability of football
- Elo ratings do not capture squad injuries, player form, or tactical matchups
- The "best 8 third-place teams" selection uses Elo as a tiebreaker — the actual FIFA criteria use points, then goal difference, then goals scored
- All probabilities are pre-tournament and do not update as games are played

---

## Author

**Jeeshan**
MBA — Finance & Analytics, University of Delhi (2026)
Open to Data Analyst, Business Analyst, and FP&A roles

[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
