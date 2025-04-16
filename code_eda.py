import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re

# Load dataset
df_weather = pd.read_csv("weather_500_project.csv")
print(df_weather.head())

# Handle missing values for numerical columns
df_weather["Sunshine"].fillna(df_weather["Sunshine"].mean(), inplace=True)
df_weather["WindGustSpeed"].fillna(df_weather["WindGustSpeed"].mean(), inplace=True)
df_weather["WindSpeed9am"].fillna(df_weather["WindSpeed9am"].mean(), inplace=True)

# Fill missing categorical values with mode
df_weather["WindGustDir"].fillna(df_weather["WindGustDir"].mode()[0], inplace=True)
df_weather["WindDir9am"].fillna(df_weather["WindDir9am"].mode()[0], inplace=True)
df_weather["WindDir3pm"].fillna(df_weather["WindDir3pm"].mode()[0], inplace=True)

# Convert categorical labels to binary
df_weather["RainToday"].replace({"Yes": 1, "No": 0}, inplace=True)
df_weather["RainTomorrow"].replace({"Yes": 1, "No": 0}, inplace=True)

# Encode direction features
df_weather["WindDir9am"] = df_weather["WindDir9am"].astype("category").cat.codes
df_weather["WindGustDir"] = df_weather["WindGustDir"].astype("category").cat.codes
df_weather["WindDir3pm"] = df_weather["WindDir3pm"].astype("category").cat.codes

print(df_weather.head())

# Display statistical summary
stats_summary = df_weather.describe()
print(stats_summary)

# Visualization: Humidity at 3 PM
plt.figure(figsize=(10, 6))
plt.bar(df_weather.index, df_weather["Humidity3pm"], color="orange")
plt.xlabel("Index")
plt.ylabel("3PM Humidity")
plt.title("Daily Humidity Levels at 3 PM")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Normalize some features
df_weather["Norm_MinTemp"] = (df_weather["MinTemp"] - df_weather["MinTemp"].min()) / (
    df_weather["MinTemp"].max() - df_weather["MinTemp"].min()
)
df_weather["Norm_MaxTemp"] = (df_weather["MaxTemp"] - df_weather["MaxTemp"].min()) / (
    df_weather["MaxTemp"].max() - df_weather["MaxTemp"].min()
)
df_weather["Norm_Humidity9am"] = (
    df_weather["Humidity9am"] - df_weather["Humidity9am"].min()
) / (df_weather["Humidity9am"].max() - df_weather["Humidity9am"].min())
df_weather["Norm_Humidity3pm"] = (
    df_weather["Humidity3pm"] - df_weather["Humidity3pm"].min()
) / (df_weather["Humidity3pm"].max() - df_weather["Humidity3pm"].min())

# Feature engineering
df_weather["Temp_Diff"] = df_weather["MaxTemp"] - df_weather["MinTemp"]
df_weather["Mean_Humidity"] = (
    df_weather["Humidity9am"] + df_weather["Humidity3pm"]
) / 2

print(
    df_weather[
        [
            "Norm_MinTemp",
            "Norm_MaxTemp",
            "Temp_Diff",
            "Norm_Humidity9am",
            "Norm_Humidity3pm",
            "Mean_Humidity",
        ]
    ].head()
)

# Scatter plot: Temperature difference vs Rainfall
plt.figure(figsize=(10, 7))
sns.scatterplot(x="Temp_Diff", y="Rainfall", data=df_weather, color="darkblue")
plt.title("Temperature Fluctuation vs Rainfall")
plt.xlabel("Temp Difference (°C)")
plt.ylabel("Rainfall (mm)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Box plot: Sunshine vs Rain Tomorrow
plt.figure(figsize=(9, 6))
sns.boxplot(x="RainTomorrow", y="Sunshine", data=df_weather, palette="Spectral")
plt.title("Sunshine Duration & Rain Forecast")
plt.xlabel("Rain Tomorrow (0=No, 1=Yes)")
plt.ylabel("Sunshine Hours")
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()

# Correlation heatmap
correlation_features = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Sunshine",
    "Humidity9am",
    "Humidity3pm",
    "Temp_Diff",
    "Mean_Humidity",
]
correlation_matrix = df_weather[correlation_features].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.7)
plt.title("Weather Feature Correlation Heatmap", fontsize=15)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Regex extract and clean wind directions
df_weather = pd.read_csv("weather_500_project.csv")
wind_north = df_weather["WindDir9am"].str.extract(r"^(N\w*)").dropna()
print(f"Directions starting with 'N':", wind_north)

df_weather["WindDir9am_Clean"] = df_weather["WindDir9am"].str.replace(
    r"[^a-zA-Z]", "", regex=True
)
print("\nSanitized WindDir9am entries:")
print(df_weather[["WindDir9am", "WindDir9am_Clean"]].head())

# Reload and handle numeric NaNs
df_weather = pd.read_csv("weather_500_project.csv")
numerical_columns = df_weather.select_dtypes(include="number").columns
df_weather[numerical_columns] = df_weather[numerical_columns].fillna(
    df_weather[numerical_columns].mean()
)

# Select features for training
input_features = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Pressure9am",
    "Pressure3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
]
X_data = df_weather[input_features]
y_labels = df_weather["RainTomorrow"].replace({"Yes": 1, "No": 0})

# Train-test split
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
    X_data, y_labels, test_size=0.3, random_state=42
)

# Model training
model_nb = GaussianNB()
model_nb.fit(X_train_set, y_train_set)
predictions = model_nb.predict(X_test_set)

# Metrics
acc = accuracy_score(y_test_set, predictions)
cmatrix = confusion_matrix(y_test_set, predictions)
class_report = classification_report(y_test_set, predictions)

print(f"Model Accuracy: {acc:.2f}")
print("\nConfusion Matrix:\n", cmatrix)
print("\nDetailed Report:\n", class_report)

# Confusion matrix heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(cmatrix, annot=True, fmt="d", cmap="coolwarm", cbar=False)
plt.title("Rain Prediction Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Interpretation
print("\nConcluding Remarks:")
print(
    f"The classifier achieved {acc:.2f} accuracy. It performs better at detecting dry days than rainy ones."
)
print(
    "Incorporating more data such as wind direction or sunshine could enhance prediction accuracy for rain."
)
