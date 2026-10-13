%%writefile app.py

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Climate Pattern Clustering",
    page_icon="🌦️",
    layout="wide"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f4f8fb;
    }

    h1 {
        color: #123b5d;
    }

    h2, h3 {
        color: #176b87;
    }

    .subtitle {
        color: #526777;
        font-size: 18px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dce6ee;
    }

    .stButton button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "climate_clustering_model.pkl"
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "The file climate_clustering_model.pkl was not found. "
            "Upload it to the same GitHub directory as app.py."
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = [
        "features",
        "imputer",
        "scaler",
        "kmeans",
        "hierarchical"
    ]

    missing_keys = [
        key for key in required_keys
        if key not in bundle
    ]

    if missing_keys:
        raise ValueError(
            f"The saved model is missing components: {missing_keys}"
        )

    return bundle


# =====================================================
# 4. APPLICATION HEADER
# =====================================================

st.title("🌦️ Climate Pattern Clustering")

st.markdown(
    '<p class="subtitle">'
    'Discover climate patterns using K-Means and '
    'Hierarchical Clustering.'
    '</p>',
    unsafe_allow_html=True
)

st.markdown("""
This application applies unsupervised machine learning to group
weather observations according to their climate characteristics.

### Machine Learning Algorithms

- **K-Means Clustering:** Groups observations around cluster centers.
- **Hierarchical Clustering:** Groups observations based on their similarity.
- **PCA:** Reduces the feature dimensions to two for visualization.
""")

st.divider()


# =====================================================
# 5. LOAD MODEL COMPONENTS
# =====================================================

try:
    model_bundle = load_model()

    features = model_bundle["features"]
    imputer = model_bundle["imputer"]
    scaler = model_bundle["scaler"]
    kmeans = model_bundle["kmeans"]

    st.success("Trained K-Means model loaded successfully.")

except Exception as error:
    st.error("Unable to load the trained machine learning model.")

    st.code(
        f"{type(error).__name__}: {error}"
    )

    st.info("""
    Check the following:
    1. climate_clustering_model.pkl exists in GitHub.
    2. The model file is in the same directory as app.py.
    3. requirements.txt includes scikit-learn==1.6.1.
    4. The model file was generated successfully in Google Colab.
    """)

    st.stop()


# =====================================================
# 6. UPLOAD DATASET
# =====================================================

st.header("📂 Upload Weather Dataset")

uploaded_file = st.file_uploader(
    "Choose a weather dataset in CSV format",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload a CSV file to start analyzing climate patterns."
    )

    st.subheader("Expected Features")

    st.write(", ".join(features))

    st.stop()


# =====================================================
# 7. READ DATASET
# =====================================================

try:
    df = pd.read_csv(uploaded_file)

except Exception as error:
    st.error("Unable to read the uploaded CSV file.")

    st.code(
        f"{type(error).__name__}: {error}"
    )

    st.stop()


if df.empty:
    st.error("The uploaded CSV file contains no records.")
    st.stop()


st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric(
        "Duplicate Records",
        int(df.duplicated().sum())
    )

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =====================================================
# 8. DATA PREPROCESSING
# =====================================================

st.header("⚙️ Data Preprocessing")

data = df.copy()

# Remove duplicate rows
data = data.drop_duplicates().reset_index(drop=True)

if len(data) < 3:
    st.error(
        "At least three unique weather observations are required."
    )
    st.stop()


# Create Month from Date if needed
if "Date" in data.columns and "Month" not in data.columns:
    parsed_dates = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data["Month"] = parsed_dates.dt.month


# Check for missing required columns
missing_columns = [
    feature for feature in features
    if feature not in data.columns
]

# Add absent features as missing values.
# The saved imputer can fill them only if it was trained
# to retain those features.
for feature in missing_columns:
    data[feature] = np.nan


# Convert features to numeric values
for feature in features:
    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce"
    )


# Prepare features in training order
X = data[features].replace(
    [np.inf, -np.inf],
    np.nan
)


# Check for features that have no usable values
all_missing = [
    feature for feature in features
    if X[feature].isna().all()
]

if all_missing:
    st.error(
        "The uploaded dataset has no valid numeric values for: "
        + ", ".join(all_missing)
    )

    st.info(
        "Upload a dataset containing the expected climate features."
    )

    st.stop()


try:
    X_imputed = imputer.transform(X)

    X_scaled = scaler.transform(X_imputed)

except Exception as error:
    st.error("Data preprocessing failed.")

    st.code(
        f"{type(error).__name__}: {error}"
    )

    st.stop()


st.success(
    f"Preprocessing completed for {len(data)} observations."
)


# =====================================================
# 9. K-MEANS CLUSTERING
# =====================================================

st.header("🔵 K-Means Clustering")

try:
    kmeans_labels = kmeans.predict(X_scaled)

    data["KMeans_Cluster"] = kmeans_labels

    st.success("K-Means clustering completed successfully.")

except Exception as error:
    st.error("K-Means clustering failed.")

    st.code(
        f"{type(error).__name__}: {error}"
    )

    st.stop()


# =====================================================
# 10. HIERARCHICAL CLUSTERING
# =====================================================

st.header("🟢 Hierarchical Clustering")

try:
    number_of_clusters = min(
        3,
        len(data)
    )

    hierarchical = AgglomerativeClustering(
        n_clusters=number_of_clusters
    )

    hierarchical_labels = hierarchical.fit_predict(
        X_scaled
    )

    data["Hierarchical_Cluster"] = hierarchical_labels

    st.success(
        "Hierarchical clustering completed successfully."
    )

except Exception as error:
    st.error("Hierarchical clustering failed.")

    st.code(
        f"{type(error).__name__}: {error}"
    )

    st.stop()


# =====================================================
# 11. CLUSTER DISTRIBUTION
# =====================================================

st.header("📈 Cluster Distribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("K-Means Distribution")

    kmeans_counts = (
        data["KMeans_Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(kmeans_counts)

    st.dataframe(
        kmeans_counts.rename("Number of Records"),
        use_container_width=True
    )

with col2:
    st.subheader("Hierarchical Distribution")

    hierarchical_counts = (
        data["Hierarchical_Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(hierarchical_counts)

    st.dataframe(
        hierarchical_counts.rename("Number of Records"),
        use_container_width=True
    )


# =====================================================
# 12. PCA VISUALIZATION
# =====================================================

st.header("🌍 Climate Cluster Visualization")

try:
    pca = PCA(
        n_components=2
    )

    reduced_data = pca.fit_transform(
        X_scaled
    )

    plot_df = pd.DataFrame({
        "PCA Component 1": reduced_data[:, 0],
        "PCA Component 2": reduced_data[:, 1],
        "K-Means Cluster": kmeans_labels,
        "Hierarchical Cluster": hierarchical_labels
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("K-Means Clusters")

        fig1, ax1 = plt.subplots(
            figsize=(7, 5)
        )

        scatter1 = ax1.scatter(
            plot_df["PCA Component 1"],
            plot_df["PCA Component 2"],
            c=plot_df["K-Means Cluster"],
            cmap="viridis",
            alpha=0.75
        )

        ax1.set_xlabel("PCA Component 1")
        ax1.set_ylabel("PCA Component 2")
        ax1.set_title("K-Means Climate Clusters")

        fig1.colorbar(
            scatter1,
            ax=ax1,
            label="Cluster"
        )

        fig1.tight_layout()

        st.pyplot(fig1)

        plt.close(fig1)

    with col2:
        st.subheader("Hierarchical Clusters")

        fig2, ax2 = plt.subplots(
            figsize=(7, 5)
        )

        scatter2 = ax2.scatter(
            plot_df["PCA Component 1"],
            plot_df["PCA Component 2"],
            c=plot_df["Hierarchical Cluster"],
            cmap="plasma",
            alpha=0.75
        )

        ax2.set_xlabel("PCA Component 1")
        ax2.set_ylabel("PCA Component 2")
        ax2.set_title("Hierarchical Climate Clusters")

        fig2.colorbar(
            scatter2,
            ax=ax2,
            label="Cluster"
        )

        fig2.tight_layout()

        st.pyplot(fig2)

        plt.close(fig2)

    st.caption(
        "PCA reduces the standardized climate features to two "
        "components for visualization. The axes do not represent "
        "individual weather measurements."
    )

except Exception as error:
    st.warning(
        "The cluster visualization could not be generated."
    )

    st.code(
        f"{type(error).__name__}: {error}"
    )


# =====================================================
# 13. CLIMATE PROFILE
# =====================================================

st.header("🌡️ Climate Profile by Cluster")

profile_data = data[
    features + ["KMeans_Cluster"]
].copy()

profile = profile_data.groupby(
    "KMeans_Cluster"
)[features].mean()

st.write(
    "Average climate characteristics for each K-Means cluster:"
)

st.dataframe(
    profile.round(2),
    use_container_width=True
)


# =====================================================
# 14. DOWNLOAD RESULTS
# =====================================================

st.header("⬇️ Download Results")

csv_output = data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Clustered Dataset",
    data=csv_output,
    file_name="climate_clustering_results.csv",
    mime="text/csv"
)


# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.caption(
    "Climate Pattern Clustering | "
    "Machine Learning Project | "
    "K-Means and Hierarchical Clustering"
)
