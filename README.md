# CupidMatch — Compatibility Prediction Engine

A machine learning pipeline that predicts relationship compatibility from personality, lifestyle, and communication traits. Built as a proof of concept using 100,000 simulated relationship pairs.

## What It Does

Takes a user profile (age, personality traits, career field, love language, etc.) and scores it against a holdout pool of 30,000+ unseen profiles to generate a personalized compatibility persona — including trait alignment, categorical likelihoods, and relationship longevity context.

## Key Findings

- **#1 predictor:** Love language alignment — 73.8% compatibility when matched vs. 35.2% when mismatched
- **#2 predictor:** Career ambition similarity — the closer two people's drive levels, the better the outcome
- When both align, compatibility rates reach **90.8%**
- Compatible pairs last **~19 months longer** on average

## Notebook Structure

| Section | Description |
|---|---|
| 1-2 | Introduction & Setup |
| 3 | Data Loading & Inspection |
| 4 | Data Cleaning, Feature Engineering (17 engineered features, OHE) |
| 5 | EDA — Top predictors, compounding effects, rescue factors |
| 6 | Preprocessing — Holdout partner pool reservation (80/20) |
| 7 | Modeling — Baseline LR, RandomizedSearchCV (5 model families), threshold tuning |
| 8 | Persona Generator — Scoring, ranges, radar chart, categorical likelihoods, longevity |
| 9 | Deliverable Export — PKL, JSON, and standalone Python engine |
| 10 | Conclusion |

## Deliverables

The notebook exports a self-contained `deliverables/` package:

```
deliverables/
  cupid_compatibility_predictor_model_option2.pkl   # Trained pipeline
  one_hot_encoder.pkl                                # Fitted encoder
  holdout_partner_pool.pkl                           # 40K individual profiles
  partner_pool_pairs.pkl                             # Pair-level holdout data
  config.json                                        # Feature columns, threshold, labels
  cupid_match_engine.py                              # Standalone engine module
```

### Usage

```python
from cupid_match_engine import generate_persona

user_profile = {
    'a_age': 29, 'a_education': 3, 'a_career_ambition': 0.70,
    'a_openness': 0.72, 'a_extraversion': 0.55, 'a_agreeableness': 0.68,
    'a_conscientiousness': 0.70, 'a_chronotype': 0.80, 'a_spontaneity': 0.35,
    'a_emotional_expressiveness': 0.55, 'a_location': 'Urban',
    'a_career_field': 'Tech', 'a_love_language': 'Quality Time',
    'a_gender': 'Female', 'a_sexual_orientation': 'Straight'
}

result = generate_persona(user_profile)
```

## Tech Stack

- Python (pandas, NumPy, scikit-learn, Plotly)
- Jupyter Notebook

## Dataset

[Cupid's Algorithm](https://www.kaggle.com/datasets/likithagedipudi/cupids-algorithm/data) — 100K synthetic relationship pairs with personality, lifestyle, and compatibility labels. Correlations designed to mimic real-world psychology patterns.

## Requirements

```
pip install pandas numpy scikit-learn plotly
```
