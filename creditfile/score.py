# Created 2023-10-18


import joblib
import lightgbm as lgb


# Artifacts
classifier = lgb.Booster(model_file='artifacts/model.txt')
scaler = joblib.load('artifacts/score-scaler.pickle')


# Scorer
def make_credit_score(features):
    'Calculate a delinquency score based on features.'
    delinquency_score = classifier.predict([features], raw_score=True)
    delinquency_score = scaler.transform([delinquency_score])[0, 0]
    credit_score = round((1-delinquency_score)*100)
    return credit_score