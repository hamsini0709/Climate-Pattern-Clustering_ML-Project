import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Climate Pattern Clustering",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Climate Pattern Clustering System")

st.markdown(
    """
    This application identifies climate patterns using
    two unsupervised machine learning algorithms:

    - K-Means Clustering
    - Hierarchical Clustering

    Upload weather observations to discover groups with
    similar temperature, humidity, rainfall, pressure,
    sunshine, and wind characteristics.
    """
)

st.divider()

# =========================================
# LOAD TRAINED MODEL
# =========================================

MODEL_PATH = "climate_clustering_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Please upload "
        "climate_clustering_model.pkl to GitHub."
    )
    st.stop()

try:
    bundle = load_model()

    features = bundle["features"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    kmeans = bundle["kmeans"]
    hierarchical = bundle["hierarchical"]

except Exception as e:
    st.error(f"Unable to load the clustering model: {e}")
    st.stop()


# =========================================
# UPLOAD DATASET
# =========================================

st.subheader("📂 Upload Weather Dataset")

uploaded_file = st.file_uploader(
    "Upload Weather_Data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        col1, col2 = st.columns(2)

        col1.metric("Total Records", data.shape[0])
        col2.metric("Total Columns", data.shape[1])

        st.subheader("Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # =========================================
        # PREPROCESS DATA
        # =========================================

        input_data = data.copy()

        if "Date" in input_data.columns:

            dates = pd.to_datetime(
                input_data["Date"],
                errors="coerce",
                dayfirst=True
            )

            input_data["Month"] = dates.dt.month

        # Validate required columns
        missing_features = [
            col for col in features
            if col not in input_data.columns
            and col != "Month"
        ]

        if missing_features:

            st.error(
                "Required climate features are missing: "
                + ", ".join(missing_features)
            )

            st.stop()

        # Ensure Month exists
        if "Month" in features and "Month" not in input_data.columns:
            st.error(
                "The dataset must contain a Date column "
                "so the Month feature can be extracted."
            )
            st.stop()

        X = input_data[features].copy()

        # Convert features to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # =========================================
        # RUN CLUSTERING
        # =========================================

        if st.button(
            "Discover Climate Patterns",
            type="primary"
        ):

            if len(X) < 3:
                st.error(
                    "Upload at least three valid records "
                    "for clustering."
                )
                st.stop()

            with st.spinner("Identifying climate patterns..."):

                # Use preprocessing fitted during training
                X_imputed = imputer.transform(X)

                X_scaled = scaler.transform(X_imputed)

                # K-Means predictions for new observations
                kmeans_labels = kmeans.predict(X_scaled)

                # Hierarchical clustering does not provide
                # a native predict() method. Therefore,
                # fit it to the uploaded observations.
                n_clusters = hierarchical.n_clusters

                if len(X) <= n_clusters:
                    st.error(
                        "Upload more records than the number "
                        "of clusters for hierarchical clustering."
                    )
                    st.stop()

                hierarchical_new = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage="ward"
                )

                hierarchical_labels = (
                    hierarchical_new.fit_predict(X_scaled)
                )

                # =========================================
                # CREATE RESULTS
                # =========================================

                results = data.copy()

                results["KMeans_Cluster"] = kmeans_labels

                results["Hierarchical_Cluster"] = (
                    hierarchical_labels
                )

                # =========================================
                # EVALUATION
                # =========================================

                km_score = silhouette_score(
                    X_scaled,
                    kmeans_labels
                )

                hc_score = silhouette_score(
                    X_scaled,
                    hierarchical_labels
                )

                st.success(
                    "Climate pattern clustering completed!"
                )

                # =========================================
                # CLUSTER SUMMARY
                # =========================================

                st.subheader("📊 Cluster Distribution")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### K-Means")

                    km_counts = (
                        results["KMeans_Cluster"]
                        .value_counts()
                        .sort_index()
                    )

                    st.bar_chart(km_counts)

                    st.metric(
                        "Silhouette Score",
                        round(km_score, 4)
                    )

                with col2:

                    st.markdown("### Hierarchical")

                    hc_counts = (
                        results["Hierarchical_Cluster"]
                        .value_counts()
                        .sort_index()
                    )

                    st.bar_chart(hc_counts)

                    st.metric(
                        "Silhouette Score",
                        round(hc_score, 4)
                    )

                # =========================================
                # CLUSTER VISUALIZATION
                # =========================================

                st.subheader("🌡️ Climate Pattern Visualization")

                fig, axes = plt.subplots(
                    1, 2,
                    figsize=(14, 5)
                )

                sns.scatterplot(
                    x=X["MaxTemp"],
                    y=X["Humidity3pm"],
                    hue=kmeans_labels,
                    palette="viridis",
                    alpha=0.7,
                    ax=axes[0]
                )

                axes[0].set_title("K-Means Clusters")
                axes[0].set_xlabel("Maximum Temperature")
                axes[0].set_ylabel("Humidity at 3 PM")

                sns.scatterplot(
                    x=X["MaxTemp"],
                    y=X["Humidity3pm"],
                    hue=hierarchical_labels,
                    palette="Set2",
                    alpha=0.7,
                    ax=axes[1]
                )

                axes[1].set_title("Hierarchical Clusters")
                axes[1].set_xlabel("Maximum Temperature")
                axes[1].set_ylabel("Humidity at 3 PM")

                st.pyplot(fig)

                plt.close(fig)

                # =========================================
                # CLUSTER PROFILES
                # =========================================

                st.subheader("📋 K-Means Climate Profiles")

                profile_data = X.copy()

                profile_data["Cluster"] = kmeans_labels

                profiles = profile_data.groupby(
                    "Cluster"
                ).mean()

                st.dataframe(
                    profiles.round(2),
                    use_container_width=True
                )

                # =========================================
                # ALGORITHM COMPARISON
                # =========================================

                st.subheader("⚖️ Algorithm Comparison")

                comparison = pd.DataFrame({
                    "Algorithm": [
                        "K-Means",
                        "Hierarchical Clustering"
                    ],
                    "Silhouette Score": [
                        km_score,
                        hc_score
                    ]
                })

                st.dataframe(
                    comparison,
                    use_container_width=True
                )

                # =========================================
                # DISPLAY RESULTS
                # =========================================

                st.subheader("📄 Final Cluster Assignments")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                # =========================================
                # DOWNLOAD RESULTS
                # =========================================

                csv_output = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Cluster Results",
                    data=csv_output,
                    file_name="climate_cluster_results.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(
            f"Unable to process the dataset: {e}"
        )

else:

    st.info(
        "Upload your weather CSV file to begin clustering."
    )

st.divider()

st.caption(
    "Climate Pattern Clustering | "
    "K-Means and Hierarchical Clustering | "
    "Unsupervised Machine Learning"
)
