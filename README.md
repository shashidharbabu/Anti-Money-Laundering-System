# 🧠 Anti-Money Laundering Detection System (Big Data Project)

This project implements a complete Big Data pipeline for detecting suspicious financial transactions using PySpark, Spark MLlib, XGBoost, and Streamlit. The system is designed to handle and process large-scale transactional datasets efficiently, apply machine learning for risk prediction, and offer an intuitive UI for both real-time and batch evaluation.

---

## 📊 Project Highlights

- **Big Data Processing:** Built using **Apache Spark (PySpark)** to scale feature engineering and labeling on millions of transactions.
- **Machine Learning Models:** Trained **GBTClassifier (MLlib)** and **XGBoost** models using engineered features.
- **Real-Time Scoring:** A user-facing **Streamlit app** allows both single transaction and batch (2GB+) scoring.
- **Batch Risk Assessment:** Large-scale transaction CSVs can be uploaded and scored with live feedback.
- **Deployment-Ready:** Includes `requirements.txt`, GitHub integration, and clean modular code for real-world deployment.

---

## 🛠️ Tech Stack

| Layer              | Tools / Libraries                            |
|--------------------|-----------------------------------------------|
| **Data Processing**| Apache Spark (PySpark)                        |
| **Feature Pipeline**| Spark SQL, window functions                  |
| **Model Training** | Spark MLlib (GBT), XGBoost                    |
| **Web UI**         | Streamlit                                     |
| **Model Management** | joblib, scikit-learn                        |
| **Version Control**| Git, GitHub                                   |
| **Data Storage**   | CSV files (2GB+ handled)                      |

---


## 🗂️ Project Structure
```


├── app.py # Streamlit app
├── requirements.txt # Dependencies
├── code/
│ ├── feature_engineering_pipeline.py
│ ├── extract_laundering_transactions.py
│ ├── xgb_aml_model.pkl
│ ├── gbt_aml_model.pkl
│ └── HI-Small_Trans.csv # Sample input
├── data/
│ └── final_ml_dataset.csv # Engineered features
└── docs/
└── architecture_diagram.png
```
---

## 🧱 Big Data Workflow Overview

### 1️⃣ Data Ingestion
- Source: IBM AML Dataset (HI-Small)
- Ingested via `SparkSession.read.csv()` with schema inference

### 2️⃣ Feature Engineering (PySpark)
- Grouped and aggregated by `From_Account`
- Features: `TxnCount`, `AvgAmount`, `StdDevAmount`, `HourBucket`, etc.
- Timestamp parsing and hour binning (`percentile_approx`)

### 3️⃣ Label Extraction
- Extract laundering patterns from semi-structured logs (`.txt`) using regex
- Join labeled transactions to feature matrix

### 4️⃣ Model Training
- Trained on Spark-generated features
- Models: `GBTClassifier` (Spark MLlib) and `XGBoostClassifier` (via joblib)

### 5️⃣ Streamlit App
- Real-time prediction: manual input form
- Batch prediction: handle 2GB+ CSVs with live progress bar
- User-selectable model for demonstration and comparison

---

## 🚀 How to Run

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

2. Run the Spark pipelines
  
3. Launch the Streamlit app
```bash
streamlit run app.py
```

🧠 Contributors
Shashidhar Babu P V D

Vaheedur Rehman Mahmud

Abhinav Vummidichetty

Suharsha
