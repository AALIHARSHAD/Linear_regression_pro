import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Linear Regression App", layout="wide")

st.title("📈 Linear Regression using Shopping Data")
st.write("Predict Spending Score using Linear Regression")

# ----------------------------------
# Load dataset
# ----------------------------------
df = pd.read_csv("Shopping_data.csv")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.write("Columns:", df.columns.tolist())

# ----------------------------------
# Drop ID column if exists
# ----------------------------------
for col in ["CustomerID", "ID"]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ----------------------------------
# Encode categorical columns
# ----------------------------------
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ----------------------------------
# Select target variable
# ----------------------------------
st.subheader("🎯 Target Selection")
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(target, axis=1)
y = df[target]

# ----------------------------------
# Train-test split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------------------
# Train Linear Regression
# ----------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------
# Prediction
# ----------------------------------
y_pred = model.predict(X_test)

# ----------------------------------
# Evaluation
# ----------------------------------
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.subheader("📊 Model Performance")
st.metric("R² Score", round(r2, 3))
st.metric("Mean Squared Error", round(mse, 3))

# ----------------------------------
# Visualization
# ----------------------------------
st.subheader("📉 Actual vs Predicted")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted Values")
st.pyplot(fig)

# ----------------------------------
# Coefficients
# ----------------------------------
st.subheader("📌 Feature Coefficients")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.dataframe(coef_df)
