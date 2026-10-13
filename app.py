import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Climate Pattern Clustering",
    page_icon="—☁—Ł",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

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


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "climate_clustering_model.pkl"
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "climate_clustering_model.pkl was not found. "
            "Please upload the trained model file to GitHub "
            "in the same folder as app.py."
        )

    return joblib.load(MODEL_PATH)


st.title("—☁—Ł Climate Pattern Clustering")

st.markdown(
    '<p class="subtitle">'
    'Discover climate patterns using K-Means and '
    'Hierarchical Clustering.'
    '</p>',
    unsafe_allow_html=True
)

st.markdown("""
This application uses unsupervised machine learning to group
weather observations according to their climate characteristics.

**Algorithms used:**
- K-Means Clustering
- Agglomerative Hierarchical Clustering
""")

st.divider()


# ==========================================
# LOAD MODEL COMPONENTS
# ==========================================

try:
    model_bundle = load_model()

    features = model_bundle["features"]
    imputer = model_bundle["imputer"]
    scaler = model_bundle["scaler"]
    kmeans = model_bundle["kmeans"]

    st.success("Trained K-Means model loaded successfully.")

except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))

    st.info(
        "Check that climate_clustering_model.pkl exists "
        "and that requirements.txt uses scikit-learn==1.6.1."
    )

    st.stop()


# ==========================================
# CSV UPLOAD
# ==========================================

st.header("└═ Upload Weather Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing weather observations",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload a CSV file to begin climate clustering."
    )

    st.subheader("Expected climate features")

    st.write(", ".join(features))

    st.stop()


# ==========================================
# READ CSV
# ==========================================

try:
    df = pd.read_csv(uploaded_file)

except Exception as error:
    st.error("Unable to read the uploaded CSV file.")
    st.code(str(error))
    st.stop()


if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()


if len(df) < 3:
    st.error(
        "Please upload a dataset containing at least "
        "3 weather observations."
    )
    st.stop()


st.subheader("└═ Dataset Preview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

st.dataframe(df.head(10), use_container_width=True)


# ==========================================
# DATA PREPROCESSING
# ==========================================

st.header("└═ Data Preprocessing")

data = df.copy()

# Remove duplicate observations
data = data.drop_duplicates().reset_index(drop=True)

if len(data) < 3:
    st.error(
        "The dataset must contain at least 3 unique observations "
        "after removing duplicates."
    )
    st.stop()


# Extract month from Date when available
if "Date" in data.columns:
    dates = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    if "Month" not in data.columns:
        data["Month"] = dates.dt.month


# Convert required features to numeric values
for feature in features:
    if feature not in data.columns:
        data[feature] = np.nan

    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce"
    )


# Prepare features in the exact training order
X = data[features].replace(
    [np.inf, -np.inf],
    np.nan
)

# Prevent failure if a required feature has no valid values
all_missing = [
    feature
    for feature in features
    if X[feature].isna().all()
]

if all_missing:
    st.warning(
        "These required features have no valid numeric values: "
        + ", ".join(all_missing)
    )

    st.error(
        "Please upload a weather dataset containing valid "
        "values for the required climate features."
    )

    st.stop()


try:
    # Apply the same preprocessing objects used during training
    X_imputed = imputer.transform(X)

    X_scaled = scaler.transform(X_imputed)

except Exception as error:
    st.error("Error during climate data preprocessing.")
    st.code(str(error))
    st.stop()


st.success(
    f"Preprocessing completed for {len(data)} observations."
)


# ==========================================
# PERFORM K-MEANS CLUSTERING
# ==========================================

st.header("└═ K-Means Clustering")

try:
    kmeans_labels = kmeans.predict(X_scaled)

    data["KMeans_Cluster"] = kmeans_labels

    st.success("K-Means clustering completed.")

except Exception as error:
    st.error("K-Means clustering failed.")
    st.code(str(error))
    st.stop()


# ==========================================
# PERFORM HIERARCHICAL CLUSTERING
# ==========================================

st.header("└═ Hierarchical Clustering")

try:
    n_clusters = min(3, len(data))

    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    hierarchical_labels = hierarchical.fit_predict(
        X_scaled
    )

    data["Hierarchical_Cluster"] = hierarchical_labels

    st.success(
        "Agglomerative hierarchical clustering completed."
    )

except Exception as error:
    st.error("Hierarchical clustering failed.")
    st.code(str(error))
    st.stop()


# ==========================================
# CLUSTER DISTRIBUTION
# ==========================================

st.header("└═ Cluster Distribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("K-Means Distribution")

    kmeans_counts = data[
        "KMeans_Cluster"
    ].value_counts().sort_index()

    st.bar_chart(kmeans_counts)

    st.dataframe(
        kmeans_counts.rename("Number of Records"),
        use_container_width=True
    )

with col2:
    st.subheader("Hierarchical Distribution")

    hierarchical_counts = data[
        "Hierarchical_Cluster"
    ].value_counts().sort_index()

    st.bar_chart(hierarchical_counts)

    st.dataframe(
        hierarchical_counts.rename("Number of Records"),
        use_container_width=True
    )


# ==========================================
# PCA VISUALIZATION
# ==========================================

st.header("└═ Climate Cluster Visualization")

try:
    pca = PCA(n_components=2)

    reduced_data = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PCA Component 1": reduced_data[:, 0],
        "PCA Component 2": reduced_data[:, 1],
        "K-Means Cluster": kmeans_labels,
        "Hierarchical Cluster": hierarchical_labels
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("K-Means Clusters")

        fig1, ax1 = plt.subplots(figsize=(7, 5))

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

        fig2, ax2 = plt.subplots(figsize=(7, 5))

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

except Exception as error:
    st.warning("The cluster visualization could not be generated.")
    st.code(str(error))


# ==========================================
# CLIMATE PROFILE
# ==========================================

st.header("└═ Climate Profile by Cluster")

profile_features = [
    feature
    for feature in features
    if feature in data.columns
]

profile_data = data[
    profile_features + ["KMeans_Cluster"]
].copy()

profile = profile_data.groupby(
    "KMeans_Cluster"
)[profile_features].mean()

st.write(
    "Average climate characteristics for each K-Means cluster:"
)

st.dataframe(
    profile.round(2),
    use_container_width=True
)


# ==========================================
# DOWNLOAD RESULTS
# ==========================================

st.header("└═ Download Results")

csv_output = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Clustered Dataset",
    data=csv_output,
    file_name="climate_clustering_results.csv",
    mime="text/csv"
)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Climate Pattern Clustering | "
    "Machine Learning Project | "
    "K-Means and Hierarchical Clustering"
)
