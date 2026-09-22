"""Prediction from current and past observations, never future labels."""
import joblib
import pandas as pd
from preprocessing import ROOT, FEATURES, feature_row


class Predictor:
    def __init__(self, path=None):
        artifact = joblib.load(path or ROOT / "models" / "gradient_boosting.joblib")
        self.model = artifact["model"]
        self.cutoff = artifact["cutoff"]

    def predict(self, supi, current_cell, previous_cell, timestamp):
        inputs = pd.DataFrame([feature_row(supi, current_cell, previous_cell, timestamp)])[FEATURES]
        probabilities = self.model.predict_proba(inputs)[0]
        classes = self.model.classes_
        index = int(probabilities.argmax())
        return {"cell": str(classes[index]), "score": float(probabilities[index]),
                "probabilities": {str(c): float(p) for c, p in zip(classes, probabilities)}}
