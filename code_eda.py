import pandas as panda
import numpy as nump
import matplotlib.pyplot as ploty
import seaborn as seab
import sklearn as skl
from sklearn.model_selection import train_test_split as train_test
from sklearn.naive_bayes import GaussianNB as Guass
from sklearn.metrics import accuracy_score as a_s
from sklearn.metrics import confusion_matrix as c_matrix
from sklearn.metrics import classification_report as c_r
import re

datafram_weather = panda.read_csv("weather_500_project.csv")
print(datafram_weather.head())

datafram_weather["Sun-shine"].fillna(datafram_weather["Sun-shine"].mean(), inplace=True)
datafram_weather["Wind-speed"].fillna(
    datafram_weather["Wind-speed"].mean(), inplace=True
)
datafram_weather["Wind-Speed-9am"].fillna(
    datafram_weather["Wind-Speed-9am"].mean(), inplace=True
)

datafram_weather["Wind-direction"].fillna(
    datafram_weather["Wind-direction"].mode()[0], inplace=True
)
datafram_weather["Wind-Dir-9am"].fillna(
    datafram_weather["Wind-Dir-9am"].mode()[0], inplace=True
)
datafram_weather["Wind-Dir-3pm"].fillna(
    datafram_weather["Wind-Dir-3pm"].mode()[0], inplace=True
)

datafram_weather["Rain-Today"].replace({"Yes": 1, "No": 0}, inplace=True)
datafram_weather["Rain-Tomorrow"].replace({"Yes": 1, "No": 0}, inplace=True)

datafram_weather["Wind-Dir-9am"] = (
    datafram_weather["Wind-Dir-9am"].astype("category").cat.codes
)
datafram_weather["Wind-direction"] = (
    datafram_weather["Wind-direction"].astype("category").cat.codes
)
datafram_weather["Wind-Dir-3pm"] = (
    datafram_weather["Wind-Dir-3pm"].astype("category").cat.codes
)

print(datafram_weather.head())

stats_summary = datafram_weather.describe()
print(stats_summary)

ploty.figure(figsize=(10, 6))
ploty.bar(datafram_weather.index, datafram_weather["Humidity-3-pm"], color="orange")
ploty.xlabel("Index")
ploty.ylabel("3PM Humidity")
ploty.title("Daily Humidity Levels at 3 PM")
ploty.xticks(rotation=45)
ploty.tight_layout()
ploty.grid(axis="y", linestyle="--", alpha=0.6)
ploty.show()

datafram_weather["Norm_MinTemp"] = (
    datafram_weather["MinT"] - datafram_weather["MinT"].min()
) / (datafram_weather["MinT"].max() - datafram_weather["MinT"].min())
datafram_weather["Norm_MaxTemp"] = (
    datafram_weather["MaxT"] - datafram_weather["MaxT"].min()
) / (datafram_weather["MaxT"].max() - datafram_weather["MaxT"].min())
datafram_weather["Norm_Humidity9am"] = (
    datafram_weather["Humidity-9-am"] - datafram_weather["Humidity-9-am"].min()
) / (datafram_weather["Humidity-9-am"].max() - datafram_weather["Humidity-9-am"].min())
datafram_weather["Norm_Humidity3pm"] = (
    datafram_weather["Humidity-3-pm"] - datafram_weather["Humidity-3-pm"].min()
) / (datafram_weather["Humidity-3-pm"].max() - datafram_weather["Humidity-3-pm"].min())

datafram_weather["Temp_Diff"] = datafram_weather["MaxT"] - datafram_weather["MinT"]
datafram_weather["Mean_Humidity"] = (
    datafram_weather["Humidity-9-am"] + datafram_weather["Humidity-3-pm"]
) / 2

print(
    datafram_weather[
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

ploty.figure(figsize=(10, 7))
seab.scatterplot(x="Temp_Diff", y="Rain-fall", data=datafram_weather, color="darkblue")
ploty.title("Temperature Fluctuation vs Rain-fall")
ploty.xlabel("Temp Difference (°C)")
ploty.ylabel("Rain-fall (mm)")
ploty.grid(True, linestyle="--", alpha=0.5)
ploty.show()

ploty.figure(figsize=(9, 6))
seab.boxplot(
    x="Rain-Tomorrow", y="Sun-shine", data=datafram_weather, palette="Spectral"
)
ploty.title("Sun-shine Duration & Rain Forecast")
ploty.xlabel("Rain Tomorrow (0=No, 1=Yes)")
ploty.ylabel("Sun-shine Hours")
ploty.grid(True, linestyle=":", alpha=0.5)
ploty.show()

correlation_features = [
    "MinT",
    "MaxT",
    "Rain-fall",
    "Sun-shine",
    "Humidity-9-am",
    "Humidity-3-pm",
    "Temp_Diff",
    "Mean_Humidity",
]
correlation_matrix = datafram_weather[correlation_features].corr()

ploty.figure(figsize=(12, 9))
seab.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.7)
ploty.title("Weather Feature Correlation Heatmap", fontsize=15)
ploty.xticks(rotation=45, ha="right")
ploty.tight_layout()
ploty.show()

datafram_weather = panda.read_csv("weather_500_project.csv")
wind_north = datafram_weather["Wind-Dir-9am"].str.extract(r"^(N\w*)").dropna()
print(f"Directions starting with 'N':", wind_north)

datafram_weather["WindDir9am_Clean"] = datafram_weather["Wind-Dir-9am"].str.replace(
    r"[^a-zA-Z]", "", regex=True
)
print("\nSanitized Wind-Dir-9am entries:")
print(datafram_weather[["Wind-Dir-9am", "WindDir9am_Clean"]].head())

datafram_weather = panda.read_csv("weather_500_project.csv")
numerical_columns = datafram_weather.select_dtypes(include="number").columns
datafram_weather[numerical_columns] = datafram_weather[numerical_columns].fillna(
    datafram_weather[numerical_columns].mean()
)

input_features = [
    "MinT",
    "MaxT",
    "Rain-fall",
    "Pressure-9-am",
    "Pressure-3-pm",
    "Wind-Speed-9am",
    "Wind-Speed-3pm",
]
X_data = datafram_weather[input_features]
y_labels = datafram_weather["Rain-Tomorrow"].replace({"Yes": 1, "No": 0})

X_train_set, X_test_set, y_train_set, y_test_set = train_test(
    X_data, y_labels, test_size=0.3, random_state=42
)

model_nb = Guass()
model_nb.fit(X_train_set, y_train_set)
predictions = model_nb.predict(X_test_set)

acc = a_s(y_test_set, predictions)
cmatrix = c_matrix(y_test_set, predictions)
class_report = c_r(y_test_set, predictions)

print(f"Model Accuracy: {acc:.2f}")
print("\nConfusion Matrix:\n", cmatrix)
print("\nDetailed Report:\n", class_report)

ploty.figure(figsize=(9, 6))
seab.heatmap(cmatrix, annot=True, fmt="d", cmap="coolwarm", cbar=False)
ploty.title("Rain Prediction Confusion Matrix", fontsize=16)
ploty.xlabel("Predicted Label")
ploty.ylabel("True Label")
ploty.show()

print("\nConcluding Remarks:")
print(
    f"The classifier achieved {acc:.2f} accuracy. It performs better at detecting dry days than rainy ones."
)
print(
    "Incorporating more data such as wind direction or Sun-shine could enhance prediction accuracy for rain."
)
