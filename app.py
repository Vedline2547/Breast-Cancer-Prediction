
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_model.pkl"

DATASET_URL = (
    "https://raw.githubusercontent.com/apogiatzis/"
    "breast-cancer-azure-ml-notebook/master/"
    "breast-cancer-data.csv"
)

FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "smoothness_mean",
    "compactness_mean"
]


# --------------------------------------------------
# 2. TRAIN MODEL
# --------------------------------------------------

def train_model():

    app.logger.info("Training breast cancer prediction model...")

    df = pd.read_csv(DATASET_URL)

    required_columns = FEATURES + ["diagnosis"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    df["diagnosis"] = df["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    df = df.dropna()

    X = df[FEATURES]
    y = df["diagnosis"].astype(int)

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=1,
        class_weight="balanced"
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    app.logger.info("Model trained and saved successfully.")

    return model


# --------------------------------------------------
# 3. LOAD OR TRAIN MODEL
# --------------------------------------------------

def load_model():

    if MODEL_PATH.exists():

        app.logger.info("Loading saved breast cancer model.")

        return joblib.load(MODEL_PATH)

    return train_model()


model = load_model()


# --------------------------------------------------
# 4. HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "home.html",
        query="",
        output1=None,
        output2=None,
        error=None
    )


# --------------------------------------------------
# 5. BREAST CANCER PREDICTION
# --------------------------------------------------

@app.route("/", methods=["POST"])
def cancerPrediction():

    try:

        input_data = {}

        # Collect and validate user inputs
        for feature in FEATURES:

            value = request.form.get(feature, "").strip()

            if not value:
                raise ValueError(
                    f"Please enter {feature.replace('_', ' ')}."
                )

            try:
                numeric_value = float(value)

            except ValueError:
                raise ValueError(
                    f"{feature.replace('_', ' ')} must be a number."
                )

            if not np.isfinite(numeric_value):
                raise ValueError(
                    f"{feature.replace('_', ' ')} must be a valid number."
                )

            if numeric_value <= 0:
                raise ValueError(
                    f"{feature.replace('_', ' ')} must be greater than zero."
                )

            input_data[feature] = numeric_value

        # Create prediction DataFrame
        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        # Generate prediction
        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        # Get confidence for the predicted class
        predicted_index = list(model.classes_).index(prediction)

        confidence = probabilities[predicted_index] * 100

        # Interpret prediction
        if prediction == 1:

            output1 = (
                "The model predicts a malignant breast cancer classification."
            )

        else:

            output1 = (
                "The model predicts a benign breast cancer classification."
            )

        output2 = f"Model confidence: {confidence:.2f}%"

        # Return results
        return render_template(
            "home.html",
            output1=output1,
            output2=output2,
            error=None,
            **{
                f"query{i + 1}": input_data[feature]
                for i, feature in enumerate(FEATURES)
            }
        )

    except ValueError as e:

        return render_template(
            "home.html",
            query="",
            error=str(e),
            output1=None,
            output2=None
        )

    except Exception:

        app.logger.exception("Prediction error")

        return render_template(
            "home.html",
            query="",
            error="An unexpected error occurred. Please try again.",
            output1=None,
            output2=None
        )


# --------------------------------------------------
# 6. APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
