# 🌫️ AQI Prediction System using Machine Learning

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=for-the-badge&logo=render)](https://aqiprediction-q6vq.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-XGBoost%20Regression-orange?style=for-the-badge&logo=scikit-learn)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/R²%20Score-71%25+-success?style=for-the-badge)](https://aqiprediction-q6vq.onrender.com)
[![Dataset](https://img.shields.io/badge/Dataset-20000%2B%20Records-informational?style=for-the-badge)](https://aqiprediction-q6vq.onrender.com)

---

## 📌 Overview

The **AQI (Air Quality Index) Prediction System** is a machine learning-powered web application that predicts air quality levels based on key environmental parameters. Built on the **Air Quality India 2015–2020** dataset from Kaggle with **20,000+ records** and a model R² score of **71%+**, this system provides reliable AQI forecasts to help users make informed decisions about outdoor activities and health safety.

🔗 **Live Demo:** [https://aqiprediction-q6vq.onrender.com](https://aqiprediction-q6vq.onrender.com)

---

## ✨ Features

- 🔮 Real-time AQI prediction using trained XGBoost ML model
- 📊 Input-based prediction with 10 environmental features
- 🌐 Deployed web interface — accessible from any device
- ⚡ Fast, lightweight, and easy to use
- 📈 71%+ R² Score on test data

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | [Air Quality India 2015–2020 — Kaggle](https://www.kaggle.com/) |
| **Total Records** | 20,000+ |
| **Time Period** | 2015 – 2020 |
| **Target Variable** | AQI (Air Quality Index) |
| **Features** | 10 environmental parameters |
| **Data Type** | Numerical / Continuous |

---

## 🧪 Input Features

The model uses the following **10 environmental features** for prediction:

| # | Feature | Description |
|---|---|---|
| 1 | **PM2.5** | Fine particulate matter (µg/m³) |
| 2 | **PM10** | Coarse particulate matter (µg/m³) |
| 3 | **NO** | Nitric oxide concentration (µg/m³) |
| 4 | **NO2** | Nitrogen dioxide concentration (µg/m³) |
| 5 | **NOx** | Total nitrogen oxides (ppb) |
| 6 | **NH3** | Ammonia concentration (µg/m³) |
| 7 | **CO** | Carbon monoxide concentration (mg/m³) |
| 8 | **SO2** | Sulfur dioxide concentration (µg/m³) |
| 9 | **O3** | Ozone concentration (µg/m³) |
| 10 | **Benzene** | Benzene concentration (µg/m³) |

> ⚠️ *Update the feature list above if your actual 10 features differ from these.*

---

## 🤖 Machine Learning

### Algorithm — XGBoost Regression

This project uses **XGBoost Regressor (Extreme Gradient Boosting)** as the core machine learning algorithm for predicting AQI values.

> XGBoost builds an ensemble of decision trees sequentially, where each new tree corrects the residual errors of the previous ones. The final AQI prediction is the weighted sum of all individual tree outputs, optimized using gradient descent.

#### Why XGBoost Regression?

| Reason | Explanation |
|---|---|
| 🚀 **High Performance** | Outperforms most traditional ML algorithms on tabular data |
| 🌿 **Handles Non-linearity** | Captures complex, non-linear relationships between pollutants and AQI |
| 🛡️ **Built-in Regularization** | L1 & L2 regularization prevents overfitting |
| ⚙️ **Feature Importance** | Automatically ranks which pollutants contribute most to AQI |
| 🔁 **Gradient Boosting** | Iteratively reduces prediction error using second-order gradients |
| ⚡ **Fast & Scalable** | Parallel tree construction handles 20,000+ records efficiently |

#### How XGBoost Works

```
Initial Prediction (mean AQI)
        ↓
   Tree 1 → Learns residual errors
        ↓
   New Prediction = Base + (learning_rate × Tree1_output)
        ↓
   Tree 2 → Learns remaining errors
        ↓
   New Prediction = Previous + (learning_rate × Tree2_output)
        ↓
        ... (repeated for n_estimators trees)
        ↓
Final AQI = Sum of all weighted tree outputs
```

#### Key Hyperparameters

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,        # Number of boosting trees
    learning_rate=0.1,       # Step size for each tree update
    max_depth=6,             # Maximum depth of each tree
    subsample=0.8,           # Fraction of data used per tree
    colsample_bytree=0.8,    # Fraction of features used per tree
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    random_state=42          # For reproducibility
)
```

#### Training Code Snippet

```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

# Load dataset
df = pd.read_csv('dataset/aqi_data.csv')

# Define features and target
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene']
X = df[features]
y = df['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Regressor
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score  : {r2_score(y_test, y_pred):.2f}")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.2f}")
```

### Model Pipeline

```
Raw Input Features (10 Environmental Parameters)
                        ↓
         Data Preprocessing & Missing Value Handling
                        ↓
           Train / Test Split (80 / 20)
                        ↓
        XGBoost Regressor — Gradient Boosted Trees
                        ↓
        Evaluation (R² Score, MAE, RMSE)
                        ↓
       Serialize Model → api_predictor.pkl (pickle)
                        ↓
         AQI Prediction & Category Output
```

### Model Performance

| Metric | Value |
|---|---|
| **R² Score** | 71%+ |
| **Algorithm** | XGBoost Regression |
| **Dataset** | Air Quality India 2015–2020 (Kaggle) |
| **Training Data** | 80% of 20,000+ records (~16,000 samples) |
| **Test Data** | 20% of 20,000+ records (~4,000 samples) |
| **Splitting Criterion** | Mean Squared Error (MSE) |

### AQI Categories

| AQI Range | Category | Health Implication |
|---|---|---|
| 0 – 50 | Good | Air quality is satisfactory |
| 51 – 100 | Moderate | Acceptable for most people |
| 101 – 150 | Unhealthy for Sensitive Groups | May affect sensitive individuals |
| 151 – 200 | Unhealthy | Everyone may experience effects |
| 201 – 300 | Very Unhealthy | Health alert for everyone |
| 301 – 500 | Hazardous | Emergency conditions |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **ML Algorithm** | XGBoost Regression |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Pickle |
| **Web Framework** | Flask |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Render |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/HackStreetBoy11/AqiPrediction.git

# 2. Navigate into the project directory
cd AqiPrediction

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

### Access the App

Open your browser and go to: `http://localhost:5000`

---

## 📁 Project Structure

```
AQI-Prediction-System-using-ML/
│
├── dataset/
│   └── aqi_data.csv            # Air Quality India 2015–2020 dataset
│
├── Model/
│   └── api_predictor.pkl       # Trained XGBoost model (pickle)
│
├── static/
│   └── css/                    # Stylesheets
│
├── templates/
│   └── index.html              # Web UI template
│
├── app.py                      # Main Flask application
├── model_training.ipynb        # Jupyter notebook for model training
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📊 How It Works

1. **User inputs** the 10 environmental/pollutant parameters into the web form.
2. The inputs are **preprocessed** using the same pipeline used during training.
3. The trained **XGBoost model predicts the AQI value**.
4. The app displays the **AQI score** along with the corresponding **health category**.

---

## 🌍 Deployment

This project is deployed on **[Render](https://render.com)** and is accessible at:

👉 **[https://aqiprediction-q6vq.onrender.com](https://aqiprediction-q6vq.onrender.com)**

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Made with ❤️ by **[Varun Sammal](https://github.com/HackStreetBoy11)**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/HackStreetBoy11)

---

> ⭐ If you found this project helpful, consider giving it a star!
