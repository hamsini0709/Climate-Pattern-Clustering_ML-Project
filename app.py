import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌦️",
    layout="wide"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "weather_prediction_model.pkl"
FEATURES_PATH = "weather_feature_names.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_names():
    if os.path.exists(FEATURES_PATH):
        return joblib.load(FEATURES_PATH)

    return None


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🌦️ Weather Prediction System")

st.markdown(
    """
    ### Predict whether it will rain tomorrow

    This machine learning application analyses historical
    weather observations and predicts the possibility of
    rainfall on the following day.

    **Model:** Gaussian Naive Bayes (GaussianNB)

    **Input:** Weather dataset in CSV format

    **Output:** Rain or No Rain prediction
    """
)

st.divider()

# ==========================================
# CHECK MODEL FILE
# ==========================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model file not found. "
        "Please upload weather_prediction_model.pkl "
        "to your GitHub repository."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# ==========================================
# GET FEATURE NAMES
# ==========================================

expected_features = None

# First, check whether the model contains feature names
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)

# If the model is a pipeline, inspect its preprocessing step
elif hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            expected_features = list(step.feature_names_in_)
            break

# If feature names are unavailable, load the saved feature list
if expected_features is None:
    expected_features = load_feature_names()


# ==========================================
# DATASET UPLOAD
# ==========================================

st.subheader("📂 Upload Weather Dataset")

uploaded_file = st.file_uploader(
    "Upload your weather CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read uploaded CSV
        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # ==========================================
        # DATASET PREVIEW
        # ==========================================

        st.subheader("📊 Uploaded Dataset Preview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Records", data.shape[0])

        with col2:
            st.metric("Total Columns", data.shape[1])

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # ==========================================
        # PREPARE INPUT DATA
        # ==========================================

        # Remove target column if it is included
        input_data = data.drop(
            columns=["RainTomorrow"],
            errors="ignore"
        ).copy()

        # Convert Date to Month if the model uses Month
        if (
            "Date" in input_data.columns
            and expected_features is not None
            and "Month" in expected_features
            and "Month" not in input_data.columns
        ):
            dates = pd.to_datetime(
                input_data["Date"],
                errors="coerce",
                dayfirst=True
            )

            input_data["Month"] = dates.dt.month

        # Remove Date if the model does not use it
        if (
            expected_features is not None
            and "Date" not in expected_features
            and "Date" in input_data.columns
        ):
            input_data = input_data.drop(
                columns=["Date"]
            )

        # ==========================================
        # VALIDATE FEATURE COLUMNS
        # ==========================================

        if expected_features is None:

            st.error(
                "The model does not contain saved feature names. "
                "Upload weather_feature_names.pkl to your "
                "GitHub repository. Create it in Colab using:\n\n"
                "joblib.dump(list(X_train.columns), "
                "'weather_feature_names.pkl')"
            )

            st.stop()

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing_features:

            st.error(
                "The uploaded dataset is missing required "
                "features: "
                + ", ".join(missing_features)
            )

            st.info(
                "Please upload the weather dataset containing "
                "the same input columns used during model training."
            )

            st.stop()

        # Keep only the required features in training order
        input_data = input_data[expected_features].copy()

        # ==========================================
        # PREDICTION
        # ==========================================

        st.subheader("🔍 Generate Predictions")

        if st.button(
            "Predict Tomorrow's Weather",
            type="primary"
        ):

            with st.spinner("Predicting weather..."):

                # Generate predictions
                predictions = model.predict(input_data)

                # Generate probabilities when supported
                probabilities = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        input_data
                    )

                # Determine the positive class index
                positive_class_index = None

                if hasattr(model, "classes_"):
                    classes = list(model.classes_)

                    if 1 in classes:
                        positive_class_index = classes.index(1)

                    elif "Yes" in classes:
                        positive_class_index = classes.index("Yes")

                # Convert model predictions into readable labels
                readable_predictions = []

                for prediction in predictions:

                    if prediction in (1, "Yes", True):
                        readable_predictions.append("Yes")
                    else:
                        readable_predictions.append("No")

                # Create results DataFrame
                results = data.copy()

                results["RainTomorrow_Prediction"] = (
                    readable_predictions
                )

                # Add probability when available
                if (
                    probabilities is not None
                    and positive_class_index is not None
                ):

                    rain_probabilities = (
                        probabilities[:, positive_class_index]
                        * 100
                    )

                    results["Rain_Probability_Percent"] = (
                        rain_probabilities.round(2)
                    )

                # ==========================================
                # DISPLAY RESULTS
                # ==========================================

                st.success("Predictions generated successfully!")

                st.subheader("🌧️ Prediction Summary")

                rain_count = readable_predictions.count("Yes")
                no_rain_count = readable_predictions.count("No")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Predictions",
                        len(readable_predictions)
                    )

                with col2:
                    st.metric(
                        "Predicted Rain",
                        rain_count
                    )

                with col3:
                    st.metric(
                        "Predicted No Rain",
                        no_rain_count
                    )

                st.subheader("📋 Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                # ==========================================
                # DOWNLOAD RESULTS
                # ==========================================

                csv_output = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv_output,
                    file_name="weather_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded file: {e}"
        )

else:

    st.info(
        "Please upload a CSV file to begin weather prediction."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Weather Prediction System | "
    "Machine Learning Project | "
    "Predictions are estimates based on historical data."
)
