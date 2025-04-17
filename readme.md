# 🌦️ Weather Data Analysis and Rain Prediction

This project involves exploring, visualizing, and modeling weather data to predict rainfall events. The goal is to apply data cleaning, feature engineering, and machine learning techniques to forecast whether it will rain the next day.

## 📁 Dataset

The dataset used is `weather_500_project.csv`, which includes historical daily weather records such as:

- Temperature (MinTemp, MaxTemp)
- Rainfall
- Sunshine hours
- Wind speed and direction
- Humidity levels
- Atmospheric pressure
- Rain indicators (RainToday, RainTomorrow)

## 🔍 Project Overview

### 1. **Data Preprocessing**

- Handled missing values using:
  - Mean for numerical columns.
  - Mode for categorical columns.
- Converted categorical wind direction into numerical codes.
- Encoded binary labels (Yes/No → 1/0).

### 2. **Feature Engineering**

- Normalized key continuous variables (MinTemp, MaxTemp, Humidity).
- Created new features:
  - `TempRange`: Difference between max and min temperature.
  - `AvgHumidity`: Average of humidity at 9AM and 3PM.

### 3. **Data Visualization**

- Bar chart for 3PM humidity trends.
- Scatter plot showing temperature variation vs rainfall.
- Box plot to explore sunshine distribution across rain predictions.
- Correlation heatmap of selected features.

### 4. **Modeling & Evaluation**

- Split data into training and testing sets (70/30).
- Trained a **Gaussian Naive Bayes classifier**.
- Evaluated model using:
  - Accuracy score
  - Confusion matrix
  - Classification report

## 📊 Key Results

- The Naive Bayes model achieved an accuracy of approximately **70%**.
- It performs better in predicting **dry days** than rainy ones.
- Suggests that additional features like sunshine hours and wind patterns could enhance the model's predictive performance.

## 🛠️ Technologies Used

- **Python 3**
- **Pandas** – Data manipulation
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn** – Machine learning and evaluation

## 📂 Project Structure

├── weather_500_project.csv # Input dataset
├── weather_analysis.py # Main script with analysis and modeling
├── README.md

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/yourusername/weather-analysis-project.git
cd weather-analysis-project
```

### Install dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### Run the script

```bash
python weather_analysis.py
```

## 📌 Future Work

- Try other models: Decision Trees, Random Forest, or XGBoost.
- Perform hyperparameter tuning and cross-validation.
- Explore external data like weather stations, seasons, or geographical zones.
