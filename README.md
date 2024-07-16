# **"Will it snow tomorrow?" - The time traveler asked**

## **1. Problem Statement**
This project aims to develop a predictive model that can determine whether it will snow tomorrow, 15 years ago from the current date. Leveraging BigQuery, climate data from over 9000 stations worldwide will be analysed. By employing data-driven techniques and machine learning models in Jupyter Notebook and VS Code, the aim is to accurately forecast the weather condition of snowfall for the specified historical date, facilitating insights into long-term weather pattern analysis.

IDE: Jupyter notebook, VS Code

## **2. How to run?**
**Step 1:** Clone the Repository  
https://github.com/Tkavwele/snow_forecasting

**Step 2:** Create and activate a virtual environment to install the dependencies   
`conda create -n myenv python==3.9`  
`conda activate myenv`  
**Step 3:** Install the requirements  

`pip install -r requirements.txt`

**Step 4:** Run this command  
`python main.py`

## **3. Workflow**
### **3.1 Data Collection**
%%bigquery 
SELECT
*,
FROM `bigquery-public-data.samples.gsod`
LIMIT 20 
* The data consists of climate information from 2005 till 2009 for station numbers including and between 725300 and 725330.
* The data is of shape `(18129, 32)`.

### **3.2 Data Preprocessing**
* **Handling Missing Values:** Columns with high percentage of missing values as well as non-informative aggregated columns are dropped. These columns include;  
```python
wban_number
mean_station_pressure
min_temperature
min_temperature_explicit
snow_depth
max_gust_wind_speed
num_mean_station_pressure_samples
num_mean_temp_samples
num_mean_dew_point_samples
num_mean_sealevel_pressure_samples
num_mean_visibility_samples
num_mean_wind_speed_samples
```
For the remaining features with fewer missing values, a `SimpleImputer` with mean strategy is used to impute the missing values in order to maintain the original statistical properties of the dataset.
* **Feature Engineering:** Categorical features such as fog and max_temperature_explicit are binary encoded. In addition, to capture patterns and seasonal variations that influence snowfall, day-of-year, month, and day-of-week features are extracted from the date features (year, month and day).
* **Feature Selection:** To improve model performance and reduce overfitting, features with perfect correlation are dropped. These include:
```python
fog
rain
hail
thunder
tornado
```
* **Feature Scaling:** A `StandardScaler` is used to standardize the range of independent features.
                     

### **3.3 Model Training**
* Different models are used for this task including, Decision Tree, Random Forest, Gradient Boosting, XGBClassifier, CatBoosting Classifier and AdaBoost Classifier.
* Grid Search hyperparameter tuning strategy is employed to optimize model performance. However, due to computational constraints, all hyperparameters could not be included in the tuning process.
* Several evaluation metrics are used, which include accuracy, precision, recall, and F1 score.
* CatBoosting Classifier had the best score based on accuracy metrics.
### **3.4 Results**
#### The Best Model: CatBoosting Classifier

**Test Accuracy:** 88.16%

**Precision:** 0.4062

**Recall:** 0.1436

**F1 Score:** 0.2122

#### **Confusion Matrix:** [[1411   38] [ 155   26]]




## **Future Scope**
* **Model Ablation Studies.** Conduct extensive ablation studies on different missing value handling methods.
* Advanced hyperparameter tuning techniques like Bayesian Optimization which leverages on probalistic models, can be used to efficiently explore and explot the hyperparameter search space.
* Model Deployment.


