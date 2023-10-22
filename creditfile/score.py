# Created 2023-10-18


from importlib_resources import files, as_file
import joblib
import lightgbm as lgb
from pathlib import Path


# Artifacts
with as_file(Path(__package__)).joinpath('artifacts/model.txt') as eml:
    classifier = lgb.Booster(model_file=eml)
with as_file(Path(__package__)).joinpath('artifacts/score-scaler.pickle') as eml:
    scaler = joblib.load(eml)


# Scorer
def make_credit_score(features):
    'Calculate a delinquency score based on features.'
    delinquency_score = classifier.predict([features], raw_score=True)
    delinquency_score = scaler.transform([delinquency_score])[0, 0]
    credit_score = round((1-delinquency_score)*100)
    return credit_score