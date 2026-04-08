# 🌫️ AQI Prediction System using Machine Learning

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=for-the-badge&logo=render)](https://aqiprediction-q6vq.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-Decision%20Tree%20Regression-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-90%25+-success?style=for-the-badge)](https://aqiprediction-q6vq.onrender.com)
[![Dataset](https://img.shields.io/badge/Dataset-2000%2B%20Records-informational?style=for-the-badge)](https://aqiprediction-q6vq.onrender.com)

---

## 📌 Overview

The **AQI (Air Quality Index) Prediction System** is a machine learning-powered web application that predicts air quality levels based on key meteorological and environmental parameters. With a dataset of **2000+ records** and a model accuracy of **90%+**, this system provides reliable AQI forecasts to help users make informed decisions about outdoor activities and health safety.

🔗 **Live Demo:** [https://aqiprediction-q6vq.onrender.com](https://aqiprediction-q6vq.onrender.com)

---

## ✨ Features

- 🔮 Real-time AQI prediction using trained ML model
- 📊 Input-based prediction with 8 meteorological features
- 🌐 Deployed web interface — accessible from any device
- ⚡ Fast, lightweight, and easy to use
- 📈 90%+ model accuracy on test data

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Total Records** | 2000+ |
| **Target Variable** | AQI (Air Quality Index) |
| **Features** | 8 meteorological parameters |
| **Data Type** | Numerical / Continuous |

---

## 🧪 Input Features

The model uses the following **8 meteorological features** for prediction:

| Feature | Symbol | Description |
|---|---|---|
| Average Temperature | **T°C** | Mean daily temperature in Celsius |
| Max Temperature | **TM°C** | Maximum recorded temperature in Celsius |
| Min Temperature | **Tm°C** | Minimum recorded temperature in Celsius |
| Sea Level Pressure | **SLP hPa** | Atmospheric pressure at sea level in hectopascals |
| Humidity | **H%** | Relative humidity percentage |
| Visibility | **VV km** | Atmospheric visibility in kilometers |
| Wind Speed | **V km/h** | Average wind speed in km/h |
| Max Wind Speed | **VM km/h** | Maximum wind speed recorded in km/h |

---

## 🤖 Machine Learning

### Algorithm — Decision Tree Regression

This project uses **Decision Tree Regression** as the core machine learning algorithm for predicting AQI values.

> A Decision Tree Regressor works by recursively splitting the dataset into subsets based on the most significant feature at each node, ultimately predicting a continuous AQI value at each leaf node.

#### Why Decision Tree Regression?

| Reason | Explanation |
|---|---|
| 🔍 **Interpretability** | Easily visualized and understood — ideal for understanding which weather factors drive AQI |
| ⚙️ **No Feature Scaling Required** | Works directly with raw numerical inputs without normalization |
| 🌿 **Non-linear Relationships** | Captures complex, non-linear relationships between weather parameters and AQI |
| 🚀 **Fast Inference** | Predictions are made in O(log n) time, making it suitable for real-time web apps |
| 📊 **Handles Mixed Data** | Works well with all 8 heterogeneous meteorological features |

#### How the Tree Works

```
                        [Root Node]
                    Is Humidity > 70%?
                   /                  \
               YES                     NO
              /                          \
   Is Wind Speed < 10?            Is Visibility > 8km?
      /        \                      /           \
  AQI: 45    AQI: 85            AQI: 120        AQI: 55
  (Good)    (Moderate)     (Unhealthy-Sensitive)  (Moderate)
```

Each internal node splits on a meteorological feature, and each leaf node outputs a predicted AQI value.

#### Key Hyperparameters

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=10,           # Controls tree depth to prevent overfitting
    min_samples_split=5,    # Minimum samples required to split a node
    min_samples_leaf=3,     # Minimum samples required at a leaf node
    random_state=42         # For reproducibility
)
```

#### Training Code Snippet

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

# Load dataset
df = pd.read_csv('dataset/aqi_data.csv')

# Define features and target
features = ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']
X = df[features]
y = df['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score  : {r2_score(y_test, y_pred):.2f}")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.2f}")
```

### Model Pipeline

```
Raw Input Features (T, TM, Tm, SLP, H, VV, V, VM)
                        ↓
         Data Preprocessing & Cleaning
                        ↓
           Train / Test Split (80 / 20)
                        ↓
     Decision Tree Regressor — Model Training
                        ↓
        Evaluation (R² Score, MAE, RMSE)
                        ↓
       Serialize Model → aqi_model.pkl (joblib)
                        ↓
         AQI Prediction & Category Output
```

### Model Performance

| Metric | Value |
|---|---|
| **Accuracy / R² Score** | 90%+ |
| **Algorithm** | Decision Tree Regression |
| **Training Data** | 80% of 2000+ records (~1600 samples) |
| **Test Data** | 20% of 2000+ records (~400 samples) |
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
| **ML Algorithm** | Decision Tree Regression (Scikit-Learn) |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Joblib |
| **Web Framework** | Flask / Streamlit |
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
│   └── aqi_data.csv            # Dataset with 2000+ records
│
├── model/
│   └── aqi_model.pkl           # Trained ML model
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

1. **User inputs** the 8 meteorological parameters into the web form.
2. The inputs are **preprocessed and scaled** using the same pipeline used during training.
3. The trained **ML model predicts the AQI value**.
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

Made with ❤️ by **[varun sammal](https://github.com/HackStreetBoy11)**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/yourusername)

---

> ⭐ If you found this project helpful, consider giving it a star!
