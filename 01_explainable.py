# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/daxs).

# COMMAND ----------

# MAGIC %md
# MAGIC # Elevator Predictive Maintenance Dataset: Anomaly Detection
# MAGIC
# MAGIC This notebook demonstrates the use of the Elevator Predictive Maintenance Dataset from Huawei German Research Center for anomaly detection. We'll use the ECOD (Empirical Cumulative Distribution Functions for Outlier Detection) algorithm from the PyOD library.
# MAGIC
# MAGIC Dataset details:
# MAGIC - Contains operation data from IoT sensors for predictive maintenance in the elevator industry.
# MAGIC - Timeseries data sampled at 4Hz during high-peak and evening elevator usage (16:30 to 23:30).
# MAGIC - Includes data from electromechanical sensors (Door Ball Bearing Sensor), ambiance (Humidity), and physics (Vibration).
# MAGIC
# MAGIC Source: [Kaggle - Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cluster setup
# MAGIC We recommend using a cluster with [Databricks Runtime 15.4 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/15.4lts-ml.html) or above. The cluster can be configured as either a single-node or multi-node CPU cluster. This notebook can run on a single-node cluster since we are applying only a single model, but the subsequent notebooks will require a multi-node cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Environment Setup
# MAGIC First, we'll set up our environment by:
# MAGIC 1. Installing required packages from requirements.txt.
# MAGIC 2. Loading utility functions from our utilities module.
# MAGIC 3. Importing necessary libraries for data analysis and modeling.

# COMMAND ----------

# DBTITLE 1,Install necessary libraries
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run a utility notebook
# MAGIC %run ./99_utilities

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pyod.models.ecod import ECOD
from sklearn.metrics import precision_score, recall_score, f1_score
from pyspark.sql.functions import current_user


# Get the current user name and store it in a variable
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
mlflow.set_experiment(f"/Users/{current_user_name}/elevator_anomaly_detection")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Storage Setup
# MAGIC We'll set up our Databricks storage environment by:
# MAGIC 1. Creating a catalog to organize our data assets.
# MAGIC 2. Setting up a schema (database) within the catalog.
# MAGIC 3. Creating a volume for CSV file storage.
# MAGIC
# MAGIC This ensures our data is properly organized and accessible.

# COMMAND ----------

catalog = "daxs"
db = "default"
volume = "csv"

# Make sure that the catalog exists
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# Make sure that the schema exists
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# Make sure that the volume exists
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{db}.{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Download
# MAGIC We'll download the Elevator Predictive Maintenance Dataset from Kaggle using [kagglehub](https://pypi.org/project/kagglehub/). The dataset will be stored in our Databricks volume for further processing.

# COMMAND ----------

import subprocess
import kagglehub

path = kagglehub.dataset_download("shivamb/elevator-predictive-maintenance-dataset", force_download=True)
bash = f"""mv {path}/predictive-maintenance-dataset.csv /Volumes/{catalog}/{db}/{volume}/predictive-maintenance-dataset.csv"""
process = subprocess.Popen(bash, shell=True, executable='/bin/bash')
process.wait()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Loading and Exploratory Data Analysis (EDA)
# MAGIC
# MAGIC Let's perform a simple analysis to understand the dataset.

# COMMAND ----------

# Load the data
df = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/predictive-maintenance-dataset.csv", header=True, inferSchema=True).toPandas()
df = df.drop(columns=["ID"])
print(f"Dataset shape: {df.shape}")
display(df.head())

# COMMAND ----------

# Basic information about the dataset
df.info()

# COMMAND ----------

# Statistical summary of the dataset
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Preprocessing and Feature Engineering
# MAGIC
# MAGIC We will apply minimal preprocessing, specifically filling missing values with -99.

# COMMAND ----------

# Handle missing values
X = df.fillna(-99)

print("Preprocessed data shape:", X.shape)
display(X.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Training and Evaluation
# MAGIC
# MAGIC ### Model Training Pipeline
# MAGIC In this section, we will:
# MAGIC 1. Split our data into training (80%) and test (20%) sets
# MAGIC 2. Train an ECOD (Empirical Cumulative Distribution Functions) model
# MAGIC 3. Log the model and its metrics using MLflow
# MAGIC 4. Register the model for future use
# MAGIC
# MAGIC The ECOD algorithm is particularly effective for anomaly detection as it:
# MAGIC - Makes no assumptions about data distribution
# MAGIC - Handles high-dimensional data well
# MAGIC - Is computationally efficient

# COMMAND ----------

# Split the data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Train the ECOD model
with mlflow.start_run(run_name="ECOD_model") as run:

    clf = ECOD(contamination=0.1, n_jobs=-1)
    clf.fit(X_train)
    clf.feature_columns_ = X_train.columns.tolist()

    # Predict on both training and test sets
    y_train_pred = clf.labels_
    y_test_pred = clf.predict(X_test)
    y_test_scores = clf.decision_function(X_test)

    # Log parameters
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("n_jobs", -1)

    # Log metrics
    train_auc = synthetic_auc(clf, X_train)
    test_auc = synthetic_auc(clf, X_test)
    mlflow.log_metric("train_auc", train_auc)
    mlflow.log_metric("test_auc", test_auc)

    # Log model
    signature = infer_signature(X_test, y_test_pred) 
    mlflow.sklearn.log_model(clf, "ecod_model", signature=signature)

    # Register the model
    model_name = f"{catalog}.{db}.ECOD_Anomaly_Detection_{current_user_name[:4]}"

    model_version = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/ecod_model", model_name)

    # Set this version as the Champion model, using its model alias
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=model_version.version
    )

    print(f"Model {model_name} version {model_version.version} is now in production")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Loading
# MAGIC In the following way you can load your champion model from MLflow registry for inference. The champion model represents our best performing version that's ready for production use.

# COMMAND ----------

# Load the champion model
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
print(f"Loaded the champion model: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Results and Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Results
# MAGIC 
# MAGIC The ECOD model detected:
# MAGIC - Training set: 8,960 anomalies (10.00% of data points)
# MAGIC - Test set: 2,212 anomalies (9.87% of data points)
# MAGIC 
# MAGIC These results align well with our configured contamination parameter of 0.1 (10%).
# MAGIC 
# MAGIC ## Model Explanations
# MAGIC 
# MAGIC Let's examine the explanations for detected anomalies. For example, in one of the most anomalous cases:
# MAGIC - The 'revolutions' sensor showed an unusually high reading of 16.933 (contributing 18% to the anomaly score)
# MAGIC - The 'x1' sensor had an extreme value of 90.132 (18% contribution)
# MAGIC - The 'x3' sensor showed an unusual reading of 0.231 (18% contribution)
# MAGIC 
# MAGIC This level of detail helps maintenance teams quickly identify which sensors are indicating potential issues.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Explanations
# MAGIC Here we generate explanations for our model's predictions using our custom explainer function.
# MAGIC The explanations will help us understand:
# MAGIC - Which features contributed most to each anomaly detection
# MAGIC - The relative importance (strength) of each feature's contribution
# MAGIC - The specific values that triggered the anomaly detection

# COMMAND ----------

explanations = explainer(clf, X_test, top_n=3)
display(explanations.sort_values('scores', ascending=False).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Conclusion and Next Steps
# MAGIC
# MAGIC In this notebook, we've performed anomaly detection on the Elevator Predictive Maintenance Dataset using the ECOD algorithm. Here are the key findings and potential next steps:
# MAGIC
# MAGIC 1. We successfully identified anomalies in both the training and test sets, with approximately 10% of data points flagged as anomalies.
# MAGIC 2. The distribution of anomaly scores shows a clear separation between normal and anomalous data points.
# MAGIC 3. We've added functionality to visualize the dimensional outlier graphs for the most and least anomalous cases in the test set, providing more insights into the nature of the anomalies.
# MAGIC
# MAGIC Next steps to improve the model and gain more insights:
# MAGIC
# MAGIC 1. Experiment with different contamination rates to fine-tune the anomaly detection threshold.
# MAGIC 2. Try other anomaly detection algorithms (e.g., Isolation Forest, Local Outlier Factor) and compare their performance.
# MAGIC 3. Perform time series analysis to identify temporal patterns in anomalies.
# MAGIC 4. Investigate the root causes of detected anomalies by analyzing the feature values of anomalous data points.
# MAGIC 5. Develop a real-time anomaly detection system for continuous monitoring of elevator performance.
# MAGIC 6. Collaborate with domain experts to validate the detected anomalies and refine the model based on their feedback.
# MAGIC 7. Analyze the dimensional outlier graphs to identify which features contribute most to the anomalies and use this information for feature selection or engineering.

# COMMAND ----------

# MAGIC %md
# MAGIC This concludes our analysis of the Elevator Predictive Maintenance Dataset using anomaly detection techniques. The insights gained from this analysis, including the visualization of the most and least anomalous cases, can be used to improve elevator maintenance strategies and reduce unplanned stops. In the following notebooks, we will explore how ECOD can be applied at scale to train thousands of models.

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries and dataset are subject to the licenses set forth below.
# MAGIC
# MAGIC | library / datas                        | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyod | A Comprehensive and Scalable Python Library for Outlier Detection (Anomaly Detection) | BSD License | https://pypi.org/project/pyod/
# MAGIC | kagglehub | Access Kaggle resources anywhere | Apache 2.0 | https://pypi.org/project/kagglehub/
# MAGIC | predictive-maintenance-dataset.csv | predictive-maintenance-dataset.csv | CC0 1.0 | https://zenodo.org/records/3653909
