import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌦️",
    layout="wide"
)

MODEL_PATH = "weather_prediction_model.pkl"
FEATURES_PATH = "weather_feature_names.pkl"


# ============================================
# LOAD MODEL AND FEATURES
# ============================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    return joblib.load(FEATURES_PATH)


# ============================================
# HEADER
# ============================================

st.title("🌦️ Weather Prediction System")

st.write(
    "Predict whether it will rain tomorrow "
    "using historical weather observations."
)

st.info(
    "This application uses a machine learning model "
    "trained on the Weather_Data.csv dataset."
)


# ============================================
# CHECK REQUIRED FILES
# ============================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Upload "
        "weather_prediction_model.pkl to GitHub."
    )
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error(
        "Feature file not found. Upload "
        "weather_feature_names.pkl to GitHub."
    )
    st.stop()

try:
    model = load_model()
    expected_features = load_features()

except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()


# ============================================
# UPLOAD DATASET
# ============================================

st.subheader("📂 Upload Weather Dataset")

uploaded_file = st.file_uploader(
    "Choose your Weather_Data.csv file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # ============================================
        # READ DATASET
        # ============================================

        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Records",
            data.shape[0]
        )

        col2.metric(
            "Total Columns",
            data.shape[1]
        )

        st.subheader("📊 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # ============================================
        # PREPARE INPUT DATA
        # ============================================

        input_data = data.drop(
            columns=["RainTomorrow"],
            errors="ignore"
        ).copy()

        # Generate Month from the original Date column
        if "Date" in input_data.columns:

            dates = pd.to_datetime(
                input_data["Date"],
                errors="coerce",
                dayfirst=True
            )

            input_data["Month"] = dates.dt.month

            input_data = input_data.drop(
                columns=["Date"]
            )

        # ============================================
        # VALIDATE INPUT FEATURES
        # ============================================

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing_features:

            st.error(
                "Missing required features: "
                + ", ".join(missing_features)
            )

            st.write(
                "Please upload the weather dataset "
                "used for training the model."
            )

            st.stop()

        # Keep the exact training feature order
        input_data = input_data[
            expected_features
        ].copy()

        # ============================================
        # PREDICTION BUTTON
        # ============================================

        st.subheader("🔍 Weather Prediction")

        if st.button(
            "Predict Tomorrow's Weather",
            type="primary"
        ):

            with st.spinner("Predicting weather..."):

                # Generate predictions
                predictions = model.predict(input_data)

                # Generate probabilities
                probabilities = model.predict_proba(
                    input_data
                )

                # Find the probability column for class 1
                classes = list(
                    model.named_steps["classifier"].classes_
                )

                positive_class_index = classes.index(1)

                rain_probabilities = (
                    probabilities[:, positive_class_index]
                    * 100
                )

                # Convert numeric predictions into labels
                labels = [
                    "Yes" if prediction == 1 else "No"
                    for prediction in predictions
                ]

                # ============================================
                # CREATE RESULTS
                # ============================================

                results = data.copy()

                results["RainTomorrow_Prediction"] = labels

                results["Rain_Probability_Percent"] = (
                    rain_probabilities.round(2)
                )

                # ============================================
                # DISPLAY SUMMARY
                # ============================================

                st.success(
                    "Predictions generated successfully!"
                )

                rain_count = labels.count("Yes")
                no_rain_count = labels.count("No")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Predictions",
                    len(labels)
                )

                col2.metric(
                    "Predicted Rain",
                    rain_count
                )

                col3.metric(
                    "Predicted No Rain",
                    no_rain_count
                )

                # ============================================
                # DISPLAY RESULTS
                # ============================================

                st.subheader("📋 Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                # ============================================
                # DOWNLOAD RESULTS
                # ============================================

                csv_output = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Predictions",
                    data=csv_output,
                    file_name="weather_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded dataset: {e}"
        )

else:

    st.info(
        "Upload Weather_Data.csv to start predicting."
    )


# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "Weather Prediction System | "
    "Random Forest Classification | "
    "Machine Learning Project"
)
