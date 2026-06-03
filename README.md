# AI-Powered Anti-Money Laundering (AML) Detection System

##  Project Overview

The AI-Powered Anti-Money Laundering (AML) Detection System is designed to automate the identification of suspicious financial transactions using Machine Learning techniques. Traditional AML processes rely heavily on manual reviews, which are time-consuming and prone to human error. This project aims to improve efficiency by analyzing transaction data and automatically flagging potentially fraudulent activities.

The system uses transaction records stored in CSV files, processes them through a trained Machine Learning model, and displays results through a user-friendly web interface built with React (Vite).

---

##  Features

* Upload transaction datasets in CSV format.
* Automatic data preprocessing and validation.
* Machine Learning-based risk prediction.
* Detection of suspicious transactions.
* Risk score generation for each transaction.
* Dashboard for transaction monitoring.
* Interactive frontend built using React and Vite.
* Fast and scalable transaction analysis.

---

##  Technology Stack

### Frontend

* React.js
* Vite
* HTML5
* CSS3
* JavaScript

### Backend

* Python
* Flask / FastAPI

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### Data Source

* CSV Transaction Dataset

---

##  Project Structure

```plaintext
AML-Detection-System/
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/
│   ├── app.py
│   ├── model.pkl
│   ├── preprocessing.py
│   └── requirements.txt
│
├── dataset/
│   └── transactions.csv
│
├── models/
│   └── trained_model.pkl
│
└── README.md
```

---

##  Working Flow

1. User uploads a CSV file containing transaction data.
2. Backend validates and preprocesses the data.
3. The trained Machine Learning model analyzes transaction patterns.
4. Risk scores are generated for each transaction.
5. Suspicious transactions are flagged.
6. Results are displayed on the dashboard.

---

##  Machine Learning Pipeline

### Data Preprocessing

* Missing value handling
* Feature selection
* Data normalization
* Encoding categorical variables

### Model Training

* Dataset preparation
* Train-test split
* Model training
* Performance evaluation

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---

## 📊 Use Cases

* Banking Institutions
* FinTech Companies
* Payment Processing Systems
* Fraud Monitoring Teams
* Compliance Departments

---

## 🔮 Future Enhancements

* Real-time transaction monitoring
* Integration with banking APIs
* Deep Learning-based anomaly detection
* Automated Suspicious Activity Reports (SAR)
* Interactive analytics dashboard
* User authentication and role management

---

##  Team Contribution

This project was developed as part of an internship to automate Anti-Money Laundering operations using AI and Machine Learning techniques. The solution focuses on reducing manual effort, improving detection speed, and enhancing compliance monitoring through intelligent transaction analysis.

---

##  License

This project is developed for educational and research purposes.
