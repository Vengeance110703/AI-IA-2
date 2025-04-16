# import python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for data visualization
import seaborn as sns  # for data visualization
import sklearn
from sklearn.model_selection import train_test_split
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load the dataset (stored in MSDA folder as weather_500_project.csv)
weather_data = pd.read_csv("weather_500_project.csv")
print(weather_data.head())  # print first few rows of the dataset

weather_data["WindDir9am"].head()

# 1. Data Preprocessing
# Handle missing values (10 Marks)
# take the mean of the column for all the numerical columns and substitute missing value with mean
weather_data["Sunshine"].fillna(weather_data["Sunshine"].mean(), inplace=True)
weather_data["WindGustSpeed"].fillna(weather_data["WindGustSpeed"].mean(), inplace=True)
weather_data["WindSpeed9am"].fillna(weather_data["WindSpeed9am"].mean(), inplace=True)

# substitute missing values with the mode for categorical columns (most frequent values)
weather_data["WindGustDir"].fillna(weather_data["WindGustDir"].mode()[0], inplace=True)
weather_data["WindDir9am"].fillna(weather_data["WindDir9am"].mode()[0], inplace=True)
weather_data["WindDir3pm"].fillna(weather_data["WindDir3pm"].mode()[0], inplace=True)

# Convert categorical variables (RainToday, RainTomorrow, WindDir9am) to numerical variables / Ordinal Encoding (10 Marks)
# replacing "yes" cells with "1" and "no" cells with "0" instead for RainToday and RainTomorrow
weather_data["RainToday"] = weather_data["RainToday"].replace({"Yes": 1, "No": 0})
weather_data["RainTomorrow"] = weather_data["RainTomorrow"].replace({"Yes": 1, "No": 0})

# Ordinally encode the WindDir9am, WindGustDir, and WindDir3pm columns (encoding the wind directions numerically)
weather_data["WindDir9am"] = weather_data["WindDir9am"].astype("category").cat.codes
weather_data["WindGustDir"] = weather_data["WindGustDir"].astype("category").cat.codes
weather_data["WindDir3pm"] = weather_data["WindDir3pm"].astype("category").cat.codes

print(weather_data.head())

# 2. Exploratory Data Analysis (EDA) (20 Marks)
# Summary statistics and dataset description (10 Marks)
summary_stats = (
    weather_data.describe()
)  # this will showcase the means, standard deviation, and more statistics in a table
print(summary_stats)

# Visualize distributions of key variables (e.g., temperature, humidity) (10 Marks)
# display the bar graph for a categorical variable (humidity at 3pm by day)
plt.figure(figsize=(10, 6))
plt.bar(weather_data.index, weather_data["Humidity3pm"], color="orange")

# Add labels and title
plt.xlabel("Day")
plt.ylabel("Humidity at 3 PM")
plt.title("Humidity Levels at 3 PM by Day")

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Adjust layout and add grid
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Display the graph
plt.show()

# 3. Feature Engineering (20 Marks)
# Normalize temperature and humidity columns manually
# formula for normalization: (x - min(x))/(max(x) - min(x))
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

# Create `TempRange` (MaxTemp - MinTemp) and `AvgHumidity` (average of Humidity9am and Humidity3pm) (15 Marks)
# `TempRange` should be 'MaxTemp' - 'MinTemp'
weather_data["TempRange"] = weather_data["MaxTemp"] - weather_data["MinTemp"]

# `AvgHumidity` is the average of Humidity at 9am and Humidity at 3pm
weather_data["AvgHumidity"] = (
    weather_data["Humidity9am"] + weather_data["Humidity3pm"]
) / 2

# Display the first few rows of the DataFrame to verify the changes
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

# 4. Advanced Visualizations (15 Marks)
# making 2 new features:
# TempRange (which will be the difference between MaxTemp and MinTemp)
# AvgHumidity (which will be the average of Humidity at 9am and 3pm)
weather_data["TempRange"] = weather_data["MaxTemp"] - weather_data["MinTemp"]
weather_data["AvgHumidity"] = (
    weather_data["Humidity9am"] + weather_data["Humidity3pm"]
) / 2

# Scatterplot: TempRange vs. Rainfall (7 Marks)
# creating a scatterplot to visualize TempRange and compare it with the amount of Rainfall
plt.figure(figsize=(10, 7))
sns.scatterplot(x="TempRange", y="Rainfall", data=weather_data, color="darkblue")
plt.title("Impact of Temperature Range on Rainfall", fontsize=14)
plt.xlabel("Daily Temperature Range (°C)", fontsize=12)
plt.ylabel("Rainfall Amount (mm)", fontsize=12)
plt.grid(True, linestyle="-.", alpha=0.5)
plt.show()

# Boxplot: Sunshine vs. RainTomorrow (8 Marks)
# comparing Sunshine hours with whether or not it rained the following day (RainTomorrow)
plt.figure(figsize=(9, 6))  # Adjusting figure size
sns.boxplot(x="RainTomorrow", y="Sunshine", data=weather_data, palette="Spectral")
plt.title("Sunshine Hours vs. Prediction of Rain Tomorrow", fontsize=14)
plt.xlabel("Will it Rain Tomorrow? (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Sunshine Duration (hours)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)  # dotted line grid
plt.show()

# 5. Correlation Analysis (10 Marks)
# Create a correlation heatmap, including new features (10 Marks)
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

# turning matrix into a heatmap
plt.figure(figsize=(12, 9))  # Adjusted the figure size slightly for better readability
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.7)

# making heading "correlation heatmap of weather features"
plt.title("Correlation Heatmap of Weather Features", fontsize=15)
plt.xticks(
    rotation=45, ha="right"
)  # tilting x-axis labels at a 45 degree angle to make it easier for people to read
plt.tight_layout()

# Show the heatmap
plt.show()

# 6. Regular Expressions (15 Marks)
# Extract wind directions starting with 'N' (5 Marks)
# first, reload the dataset since I overwrote WindDir9am column data with numeric values in the first part
weather_data = pd.read_csv("weather_500_project.csv")

# find out what wind directions begin with 'N'
winddirect_firstletter_n = weather_data["WindDir9am"].str.extract(r"^(N\w*)")
winddirect_firstletter_n = (
    winddirect_firstletter_n.dropna()
)  # do this to remove rows that don’t match the regex pattern (values that don't start with 'N', and null/NAN values)

# Display the extracted values that start with 'N'
print(f"Wind directions that begin with the letter 'N': ", winddirect_firstletter_n)

# Clean WindDir9am column using regex (10 Marks)
# using regex to remove any non-alphabetic characters or whitespace from the WindDir9am column
weather_data["WindDir9am_cleaned"] = weather_data["WindDir9am"].str.replace(
    r"[^a-zA-Z]", "", regex=True
)

# show the first few rows of the cleaned WindDir9am column
print("\nCleaned WindDir9am column:")
print(weather_data[["WindDir9am", "WindDir9am_cleaned"]].head())

# reloading the dataset
weather_data = pd.read_csv("weather_500_project.csv")

# filling in missing values instead of dropping them
# i will use the mean for missing values so that no data gets lost
# Fill only numeric columns with their mean
numeric_cols = weather_data.select_dtypes(include="number").columns
weather_data[numeric_cols] = weather_data[numeric_cols].fillna(
    weather_data[numeric_cols].mean()
)


# selecting features: 'Pressure9am' and 'Pressure3pm' instead of 'Sunshine'
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
y = weather_data["RainTomorrow"].replace(
    {"Yes": 1, "No": 0}
)  # encoding 'RainTomorrow' as 1 (Yes) and 0 (No)

# splitting the data into training and testing sets, keeping 30% for testing and set a random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# training the naive bayes classifier on the training data
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# making predictions on the test data
y_pred = nb_model.predict(X_test)

# evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# printing the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# visualizing the confusion matrix with a heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=False)
plt.title("Confusion Matrix: Rain Prediction", fontsize=16)
plt.xlabel("Predicted Outcome", fontsize=12)
plt.ylabel("Actual Outcome", fontsize=12)
plt.show()

# final insights and analysis
print("\nAnalysis & Insights:")
print(
    f"With an accuracy of {accuracy:.2f}, the model seems to do well in predicting when it won't rain(no), but not as well with predicting rain (yes)."
)
print(
    "Maybe adding other features like wind direction or sunshine could improve the model's accuracy, especially for rainy day predictions."
)
