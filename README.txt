# PG-DBDA_Project
Churn Prediction and Cohorts analysis



Cohort Analysis:-
Let's go through the code step by step and explain each part in detail:

**Step 1: Importing Libraries**

import pandas as pd
import numpy as np

This step imports the necessary libraries: `pandas` for data manipulation and analysis, and `numpy` for numerical operations.

**Step 2: Loading Datasets**

df_address = pd.read_csv('df_address.csv')
df_customer = pd.read_csv('df_customer.csv')
df_demographic = pd.read_csv('df_demographic.csv')
merged_df = pd.read_csv("autoinsurance.csv")

In this step, four datasets (`df_address`, `df_customer`, `df_demographic`, `merged_df`) are loaded from CSV files into Pandas DataFrames.

**Step 3: Merging Datasets**

merge = pd.merge(df_demographic, df_customer, on='INDIVIDUAL_ID', how='inner')
merged_df["INDIVIDUAL_ID"] = merged_df["Customer ID"]
merged_df.drop("Customer ID", axis=1, inplace=True)
merge_F = pd.merge(merge, merged_df, on='INDIVIDUAL_ID', how='inner')

Datasets are merged using the `INDIVIDUAL_ID` column as the primary key for INNER JOIN operations.

**Step 4: Data Preprocessing**

inner_merged["InvoiceDate"] = pd.to_datetime(inner_merged["InvoiceDate"])
inner_merged["InvoiceMonth"] = inner_merged["InvoiceDate"].apply(get_month)
inner_merged["YEAR"] = inner_merged['InvoiceDate'].dt.year
inner_merged["CohortMonth"] = inner_merged.groupby("INDIVIDUAL_ID")["InvoiceMonth"].transform("min")

Date columns are converted to datetime format, and new columns like `InvoiceMonth`, `YEAR`, and `CohortMonth` are created for cohort analysis.

**Step 5: Cohort Analysis**

cohort_1 = Create_cohorts(data_2009_2010)
cohort_2 = Create_cohorts(data_2010_2011)

Cohorts are created using the `Create_cohorts` function, which calculates the cohort index and cohort month based on the acquired month of each customer.

**Step 6: Visualization**

Cohort_heatmap(retention_1)

Heatmap visualizations are created to display cohort retention rates using the `Cohort_heatmap` function.

**Step 7: Customer Analysis**

fig, ax = plt.subplots(figsize=(15, 7), facecolor="#adad85")
sns.lineplot(x=inner_merged["InvoiceMonth"].unique(), y=customers, marker='o', color='g', label='New Customers')
sns.lineplot(x=inner_merged["InvoiceMonth"].unique(), y=customer_in_month, marker='o', color='r', label='Re-visiting Customers')

Line charts are used to visualize new and revisiting customers per month.

**Step 8: Time Series Predictions**

plot_series(y_train, y_test, labels=["y_train", "y_test"])

Time series data is prepared for forecasting and plotted using the `plot_series` function.

**Step 9: Machine Learning: Churn Prediction**

inner_merged_1 = inner_merged.drop(["ADDRESS_ID", "INDIVIDUAL_ID", ..., "CohortIndex"], axis=1)
inner_merged_1 = pd.get_dummies(inner_merged_1, drop_first=True)

Unnecessary columns are dropped, and categorical variables are encoded using one-hot encoding.

**Step 10: Exporting Results**

inner_merged_1.to_csv("Churn_Predictn.csv", index=False)

The final preprocessed dataset is saved as a CSV file named "Churn_Predictn.csv".



Churn_prediction_explanation:-
This code uses various classification algorithms (Decision Tree, Random Forest, XGBoost, and Logistic Regression) to predict customer churn. Churn prediction involves determining whether customers are likely to stop using a service or product. Let's go through the code step by step:

1. **Import Libraries:** The necessary libraries are imported for data manipulation, visualization, and machine learning model training and evaluation.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


2. **Read Data:** The code reads a CSV file named "Churn_Predictn.csv" into a pandas DataFrame called `df`.


df = pd.read_csv("Churn_Predictn.csv")


3. **Data Exploration:**

   - `df.dtypes` displays the data types of each column.
   - `df.isna().sum()` calculates the sum of missing values in each column.

4. **Train-Test Split:**

   The data is split into training and testing sets using a 75-25 split ratio. The target variable for prediction is "CHURN".

train, test = train_test_split(df, test_size=0.25, stratify=df["CHURN"], random_state=23)


5. **Exploratory Data Analysis:**

   - `train.CHURN.value_counts(normalize=True)*100` calculates the percentage distribution of churn values in the training set.
   - `train.shape` and `test.shape` display the number of rows and columns in the training and testing sets.
   - A histogram of the "CHURN" column is plotted using Seaborn and Matplotlib.


x = train.drop("CHURN", axis=1)
y = train["CHURN"]
sns.set_style("darkgrid")
sns.histplot(y.astype(str))
plt.show()


6. **Stratified K-Fold Cross-Validation:**

   `StratifiedKFold` is used to perform stratified K-fold cross-validation. It ensures that each fold maintains the same proportion of target classes as the original dataset.


kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=23)


7. **Model Initialization and Hyperparameter Tuning:**

   Four classification models are used: Decision Tree, Random Forest, Logistic Regression, and XGBoost. For each model:
   
   - The model's default parameters are displayed using `model.get_params()`.
   - Hyperparameters are defined in a dictionary to be tuned using `GridSearchCV`.
   - `GridSearchCV` performs grid search with cross-validation to find the best hyperparameters for each model based on negative log loss.
   
8. **Model Fitting and Best Parameters Display:**

   The four models are fitted to the training data using the best hyperparameters found by grid search. The best parameters and the corresponding best log loss scores are printed for each model.

9. **Model Evaluation on Test Data:**

   The trained models are evaluated on the test data by making predictions and calculating accuracy scores.

10. **Saving the Best Model:**

    The best performing Random Forest model is saved using the `pickle` library. The model is serialized and saved as a binary file named "churn_prediction_model.sav".

11. **Loading the Saved Model:**

    The saved Random Forest model is loaded back into memory.

12. **Feature Importance Visualization:**

    The feature importances of the loaded Random Forest model are extracted and sorted. The importance values are then plotted in a horizontal bar chart to visualize the importance of each feature.

This code demonstrates the process of loading data, preprocessing, model selection, hyperparameter tuning, model evaluation, and feature importance visualization in a churn prediction scenario.

Web_app_explanation:-
Let's go through the provided code step by step and explain what each part is doing:

**Step 1: Importing Libraries**

import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

In this step, necessary libraries are imported: `numpy` for numerical operations, `pickle` for loading the saved machine learning model, `streamlit` for creating web applications, `matplotlib.pyplot` for creating plots, and `seaborn` for enhancing plot aesthetics.

**Step 2: Loading the Saved Model**

churn_model = pickle.load(open("C:/DBDA/Project/dataset/churn_prediction_model.sav", 'rb'))

Here, a previously trained machine learning model for churn prediction is loaded using the `pickle` module.

**Step 3: Churn Prediction Function**

def churn_prediction(input_data):
    input_data_as_numpy_array = np.array(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = churn_model.predict(input_data_reshaped)
    
    if prediction[0] == 1:
        return 'The person may churn'
    else:
        return 'The person may not churn'

This function takes input data as a list of features, converts it to a NumPy array, reshapes it for prediction, and then predicts whether the person will churn or not based on the loaded model.

**Step 4: Main Web Application**

def main():
    st.title('Churn Prediction Web app')
    # ... (user input section)

    if st.button('Predict Chrun'):
        try:
            # ... (prediction and visualization)
        except:
            st.error("An error occurred during prediction. Please check your inputs.")
    
    st.success(churn)

if __name__ == '__main__':
    main()

The `main()` function sets up the main web application. It gives a title to the app, collects user input (ordinal features and dummy features), and provides a button to predict churn. If the "Predict Churn" button is clicked, it tries to predict the churn status using the `churn_prediction()` function and displays the prediction result along with a pie chart showing the predicted probabilities. If an error occurs during prediction, it displays an error message.

The `if __name__ == '__main__':` block ensures that the `main()` function is executed when the script is run directly.

**Step 5: Dummy Feature Encoding**

if HOME_OWNER == 'Yes':
    HOME_OWNER_dum = 1
else:
    HOME_OWNER_dum = 0
    
# (similar encoding for other dummy features)

if HOME_MARKET_VALUE == "25000 - 49999":
    HOME_MARKET_VALUE_dum = [0, 0, ..., 0, 0]
elif HOME_MARKET_VALUE == "50000 - 74999":
    HOME_MARKET_VALUE_dum = [0, 0, ..., 0, 0]
# ... (similar encoding for other ranges)
else:
    HOME_MARKET_VALUE_dum = [0, 1, ..., 0, 0]

This section encodes the categorical dummy features into numerical format. Each dummy variable is assigned a binary value (0 or 1) based on the user's selection.

**Step 6: Predicting Churn and Displaying Results**

if st.button('Predict Chrun'):
    try:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        churn = churn_prediction(features)
        probab = churn_model.predict_proba(np.array(features).reshape(1, -1))
        
        st.success(churn)
        labels = ['Churn', 'Not Churn']
        plt.figure(figsize=(3, 3))
        plt.pie(probab[0], labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Predicted Probabilities')
        st.pyplot()
    except:
        st.error("An error occurred during prediction. Please check your inputs.")

When the "Predict Churn" button is clicked, the code attempts to predict churn using the `churn_prediction()` function. If successful, it displays the prediction result (whether the person may churn or not) using `st.success()`. It also calculates and displays a pie chart with predicted probabilities of churn using `st.pyplot()`.

In case of an error during prediction, it displays an error message using `st.error()`.

**Step 7: Running the Main Function**

if __name__ == '__main__':
    main()

This ensures that the `main()` function is executed when the script is run directly, effectively running the entire web application.

In summary, the provided code creates a web application for predicting churn based on user-provided features. Users input various features through the Streamlit interface, and the application uses a pre-trained machine learning model to predict whether the individual may churn or not. The prediction result is displayed along with a pie chart showing the predicted probabilities.






