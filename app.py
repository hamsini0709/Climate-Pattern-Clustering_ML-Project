import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Weather Prediction",
    page_icon="🌦️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("weather_prediction_model.pkl")

model = load_model()

st.title("🌦️ Weather Prediction System")

st.write(
    "Upload weather observations to predict "
    "whether it will rain tomorrow."
)

st.info(
    "This application uses a machine learning model "
    "trained on historical weather data. "
    "Predictions are estimates, not official forecasts."
)

st.subheader("Upload Weather Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.write("Uploaded dataset preview:")
        st.dataframe(data.head())

        # Remove target if it is included in the uploaded file
        input_data = data.drop(
            columns=["RainTomorrow"],
            errors="ignore"
        ).copy()

        # Extract month from Date when provided
        if "Date" in input_data.columns:
            dates = pd.to_datetime(
                input_data["Date"],
                errors="coerce"
            )

            input_data["Month"] = dates.dt.month

            input_data = input_data.drop(
                columns=["Date"]
            )

        # Check required model features
        expected_features = list(
            model.feature_names_in_
        )

        missing_features = [
            col for col in expected_features
            if col not in input_data.columns
        ]

        if missing_features:

            st.error(
                "The uploaded CSV is missing required columns: "
                + ", ".join(missing_features)
            )

        else:

            # Keep columns in the expected order
            input_data = input_data[
                expected_features
            ]

            # Generate predictions
            predictions = model.predict(
                input_data
            )

            probabilities = model.predict_proba(
                input_data
            )[:, 1]

            results = data.copy()

            results["RainTomorrow_Prediction"] = [
                "Yes" if value == 1 else "No"
                for value in predictions
            ]

            results["Rain_Probability_Percent"] = (
                probabilities * 100
            ).round(2)

            st.subheader("Prediction Results")

            st.dataframe(results)

            rain_count = int(
                (predictions == 1).sum()
            )

            no_rain_count = int(
                (predictions == 0).sum()
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Rain",
                rain_count
            )

            col2.metric(
                "Predicted No Rain",
                no_rain_count
            )

            csv_output = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions",
                data=csv_output,
                file_name="weather_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded file: {e}"
        )
