import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🧠 Customer Segmentation using K-Means")
st.write("Upload a customer dataset and perform K-Means clustering.")

# ----------------------------------
# File upload
# ----------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to continue")
    st.stop()

# ----------------------------------
# Load data
# ----------------------------------
df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.write("Columns:", df.columns.tolist())

# ----------------------------------
# Data Cleaning
# ----------------------------------
st.subheader("🧹 Data Cleaning")

# Drop ID if present
if "ID" in df.columns:
    df.drop("ID", axis=1, inplace=True)

# Separate numeric & categorical columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# Fill missing values
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

st.success("Missing values handled successfully")

# ----------------------------------
# Encoding (for K-Means only)
# ----------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

# ----------------------------------
# Scaling
# ----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# ----------------------------------
# Choose K
# ----------------------------------
st.subheader("🔢 Select Number of Clusters")
k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)

# ----------------------------------
# Train K-Means
# ----------------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ----------------------------------
# Evaluation
# ----------------------------------
sil_score = silhouette_score(X_scaled, clusters)
st.metric("Silhouette Score", round(sil_score, 3))

# ----------------------------------
# Visualization (SAFE)
# ----------------------------------
st.subheader("📊 Cluster Visualization")

# Choose columns for plotting
x_col = st.selectbox("X-axis", num_cols)
y_col = st.selectbox("Y-axis", num_cols, index=1 if len(num_cols) > 1 else 0)

fig, ax = plt.subplots()
sns.scatterplot(
    x=x_col,
    y=y_col,
    hue="Cluster",
    data=df,
    palette="Set2",
    ax=ax
)
ax.set_title("Customer Segmentation using K-Means")
st.pyplot(fig)

# ----------------------------------
# Cluster Summary
# ----------------------------------
st.subheader("📌 Cluster Summary (Mean Values)")
summary = df.groupby("Cluster")[num_cols].mean()
st.dataframe(summary)

# ----------------------------------
# Download Result
# ----------------------------------
st.subheader("⬇️ Download Clustered Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="clustered_customers.csv",
    mime="text/csv"
)

