from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Load saved model package
model_path = Path("models/final_model.pkl")
model_package = joblib.load(model_path)

model = model_package["model"]
scaler = model_package["scaler"]
feature_columns = model_package["feature_columns"]

numeric_features = model_package["numeric_features"]


def prepare_input(data):
    """
    Convert user input into the same format
    used during model training.
    """

    df = pd.DataFrame([data])

    # Convert categorical variables using one-hot encoding
    df = pd.get_dummies(
        df,
        columns=["Gender", "International"],
        drop_first=True
    )

    # Make sure the input has exactly the same columns
    # as the training data.
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale the numerical features
    df[numeric_features] = scaler.transform(
        df[numeric_features]
    )

    return df


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        student_data = {
            "Age at enrollment": float(
                request.form["age"]
            ),

            "Admission grade": float(
                request.form["admission_grade"]
            ),

            "Previous qualification (grade)": float(
                request.form["previous_grade"]
            ),

            "Scholarship holder": int(
                request.form["scholarship"]
            ),

            "Tuition fees up to date": int(
                request.form["tuition"]
            ),

            "Debtor": int(
                request.form["debtor"]
            ),

            "Gender": int(
                request.form["gender"]
            ),

            "International": int(
                request.form["international"]
            )
        }

        input_data = prepare_input(student_data)

        prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)