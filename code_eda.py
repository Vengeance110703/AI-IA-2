import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

weather_data = pd.read_csv("weather_500_project.csv")
print(weather_data.head())

weather_data["WindDir9am"].head()

weather_data["Sunshine"].fillna(weather_data["Sunshine"].mean(), inplace=True)
weather_data["WindGustSpeed"].fillna(weather_data["WindGustSpeed"].mean(), inplace=True)
weather_data["WindSpeed9am"].fillna(weather_data["WindSpeed9am"].mean(), inplace=True)

weather_data["WindGustDir"].fillna(weather_data["WindGustDir"].mode()[0], inplace=True)
weather_data["WindDir9am"].fillna(weather_data["WindDir9am"].mode()[0], inplace=True)
weather_data["WindDir3pm"].fillna(weather_data["WindDir3pm"].mode()[0], inplace=True)

weather_data["RainToday"] = weather_data["RainToday"].replace({"Yes": 1, "No": 0})
weather_data["RainTomorrow"] = weather_data["RainTomorrow"].replace({"Yes": 1, "No": 0})

weather_data["WindDir9am"] = weather_data["WindDir9am"].astype("category").cat.codes
weather_data["WindGustDir"] = weather_data["WindGustDir"].astype("category").cat.codes
weather_data["WindDir3pm"] = weather_data["WindDir3pm"].astype("category").cat.codes

print(weather_data.head())

summary_stats = weather_data.describe()
print(summary_stats)

plt.figure(figsize=(10, 6))
plt.bar(weather_data.index, weather_data["Humidity3pm"], color="orange")
plt.xlabel("Day")
plt.ylabel("Humidity at 3 PM")
plt.title("Humidity Levels at 3 PM by Day")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

weather_data["MinTemp_normalized"] = (
    weather_data["MinTemp"] - weather_data["MinTemp"].min()
) / (weather_data["MinTemp"].max() - weather_data["MinTemp"].min())
weather_data["MaxTemp_normalized"] = (
    weather_data["MaxTemp"] - weather_data["MaxTemp"].min()
) / (weather_data["MaxTemp"].max() - weather_data["MaxTemp"].min())
weather_data["Humidity9am_normalized"] = (
    weather_data["Humidity9am"] - weather_data["Humidity9am"].min()
) / (weather_data["Humidity9am"].max() - weather_data["Humidity9am"].min())
weather_data["Humidity3pm_normalized"] = (
    weather_data["Humidity3pm"] - weather_data["Humidity3pm"].min()
) / (weather_data["Humidity3pm"].max() - weather_data["Humidity3pm"].min())

weather_data["TempRange"] = weather_data["MaxTemp"] - weather_data["MinTemp"]
weather_data["AvgHumidity"] = (
    weather_data["Humidity9am"] + weather_data["Humidity3pm"]
) / 2

print(
    weather_data[
        [
            "MinTemp_normalized",
            "MaxTemp_normalized",
            "TempRange",
            "Humidity9am_normalized",
            "Humidity3pm_normalized",
            "AvgHumidity",
        ]
    ].head()
)

weather_data["TempRange"] = weather_data["MaxTemp"] - weather_data["MinTemp"]
weather_data["AvgHumidity"] = (
    weather_data["Humidity9am"] + weather_data["Humidity3pm"]
) / 2

plt.figure(figsize=(10, 7))
sns.scatterplot(x="TempRange", y="Rainfall", data=weather_data, color="darkblue")
plt.title("Impact of Temperature Range on Rainfall", fontsize=14)
plt.xlabel("Daily Temperature Range (°C)", fontsize=12)
plt.ylabel("Rainfall Amount (mm)", fontsize=12)
plt.grid(True, linestyle="-.", alpha=0.5)
plt.show()

plt.figure(figsize=(9, 6))
sns.boxplot(x="RainTomorrow", y="Sunshine", data=weather_data, palette="Spectral")
plt.title("Sunshine Hours vs. Prediction of Rain Tomorrow", fontsize=14)
plt.xlabel("Will it Rain Tomorrow? (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Sunshine Duration (hours)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

corr_matrix = weather_data[
    [
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Sunshine",
        "Humidity9am",
        "Humidity3pm",
        "TempRange",
        "AvgHumidity",
    ]
].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.7)
plt.title("Correlation Heatmap of Weather Features", fontsize=15)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

weather_data = pd.read_csv("weather_500_project.csv")
winddirect_firstletter_n = weather_data["WindDir9am"].str.extract(r"^(N\w*)")
winddirect_firstletter_n = winddirect_firstletter_n.dropna()
print(f"Wind directions that begin with the letter 'N': ", winddirect_firstletter_n)

weather_data["WindDir9am_cleaned"] = weather_data["WindDir9am"].str.replace(
    r"[^a-zA-Z]", "", regex=True
)
print("\nCleaned WindDir9am column:")
print(weather_data[["WindDir9am", "WindDir9am_cleaned"]].head())

weather_data = pd.read_csv("weather_500_project.csv")
numeric_cols = weather_data.select_dtypes(include="number").columns
weather_data[numeric_cols] = weather_data[numeric_cols].fillna(
    weather_data[numeric_cols].mean()
)

features = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Pressure9am",
    "Pressure3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
]
X = weather_data[features]
y = weather_data["RainTomorrow"].replace({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

plt.figure(figsize=(9, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=False)
plt.title("Confusion Matrix: Rain Prediction", fontsize=16)
plt.xlabel("Predicted Outcome", fontsize=12)
plt.ylabel("Actual Outcome", fontsize=12)
plt.show()

print("\nAnalysis & Insights:")
print(
    f"With an accuracy of {accuracy:.2f}, the model seems to do well in predicting when it won't rain(no), but not as well with predicting rain (yes)."
)
print(
    "Maybe adding other features like wind direction or sunshine could improve the model's accuracy, especially for rainy day predictions."
)
