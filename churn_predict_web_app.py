import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

#loading the saved model
churn_model = pickle.load(open("C:/DBDA/Project/dataset/churn_prediction_model.sav",'rb'))

def churn_prediction(input_data):
    #changing the input_data to numpy array
    #input_data_as_numpy_array = np.asarray(input_data)
    input_data_as_numpy_array = np.array(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = churn_model.predict(input_data_reshaped)
    
    if (prediction[0]==1):
        return 'The person may not churn'
    
    else:
        return 'The person may churn'
    
def main():
    #giving a title
    st.title('Churn Prediction Web app')
    
    #getting the input data from the user
    
    #Ordinal features
    INCOME = st.number_input('Income ($)', min_value = 0)
    HAS_CHILDREN = st.number_input('No. of children', min_value = 0)
    LENGTH_OF_RESIDENCE = st.number_input('Length of residence in years', min_value = 0)
    CURR_ANN_AMT = st.number_input('Annual Installment', min_value = 100)
    DAYS_TENURE = st.number_input('Tenure in days', min_value = 29)
    AGE_IN_YEARS = st.number_input('Age (years)', min_value = 18)
    
    #Dummies start from here
    HOME_OWNER = st.selectbox('Home Owner?',['Yes','No']) #Female,Male
    COLLEGE_DEGREE = st.selectbox('Posses College Degree ?',['No','Yes']) #No,Yes
    GOOD_CREDIT = st.selectbox('Has Good Credit?',['No','Yes']) #No,Yes
    MARITAL_STATUS_Single = st.selectbox('Ever Married?',['No','Yes']) #No,Yes
    HOME_MARKET_VALUE = st.selectbox('Market Value of House ($)?', ["25000 - 49999","50000 - 74999","75000 - 99999","100000 - 124999",\
                                                                    "125000 - 149999", "150000 - 174999","175000 - 199999","200000 - 224999","225000 - 249999",\
                                                                    "250000 - 274999", "275000 - 299999", "300000 - 349999", "350000 - 399999", "400000 - 449999",\
                                                                     "450000 - 499999", "500000 - 749999", "750000 - 999999","1000000 Plus"]
) 
        
    #Encoding the dummy features
    if HOME_OWNER == 'Yes':
        HOME_OWNER_dum=1
    else:
        HOME_OWNER_dum = 0
        
    if COLLEGE_DEGREE == 'No':
        COLLEGE_DEGREE_dum=0
    else:
        COLLEGE_DEGREE_dum=1
    
    if GOOD_CREDIT == 'No':
        GOOD_CREDIT_dum=0
    else:
        GOOD_CREDIT_dum=1
        
    if MARITAL_STATUS_Single == 'No':
        MARITAL_STATUS_Single_dum=1
    else:
        MARITAL_STATUS_Single_dum=0
    
    #Government Job,Never worked,Private Job,Self employed,Children
    if HOME_MARKET_VALUE == "25000 - 49999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "50000 - 74999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    elif HOME_MARKET_VALUE == "75000 - 99999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    elif HOME_MARKET_VALUE == "100000 - 124999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "125000 - 149999":
        HOME_MARKET_VALUE_dum =[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "150000 - 174999":
        HOME_MARKET_VALUE_dum =[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "175000 - 199999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "200000 - 224999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "225000 - 249999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "250000 - 274999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "275000 - 299999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "300000 - 349999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "350000 - 399999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "400000 - 449999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    elif HOME_MARKET_VALUE == "450000 - 499999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
    elif HOME_MARKET_VALUE == "500000 - 749999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
    elif HOME_MARKET_VALUE == "750000 - 999999":
        HOME_MARKET_VALUE_dum =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    else:
        HOME_MARKET_VALUE_dum =[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
   
        
    # code for Prediction
    churn = ''
    
    #Defining list of features
    features = [INCOME, HAS_CHILDREN, LENGTH_OF_RESIDENCE, HOME_OWNER_dum,
       COLLEGE_DEGREE_dum, GOOD_CREDIT_dum, CURR_ANN_AMT, DAYS_TENURE,
       AGE_IN_YEARS, MARITAL_STATUS_Single_dum] + HOME_MARKET_VALUE_dum
                    
    
        
    # creating a button for Prediction
    
    if st.button('Predict Chrun'):
        try: 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            churn = churn_prediction(features)
            probab = churn_model.predict_proba(np.array(features).reshape(1,-1))

            st.success(churn)
            labels = ['Not Churn', 'Churn']
            plt.figure(figsize=(3,4))
            
            plt.pie(probab[0], labels=labels, autopct='%1.1f%%', startangle=140)
            plt.title('Predicted Probabilities')
            st.pyplot()
        except:
            st.error("An error occurred during prediction. Please check your inputs.")
        
    st.success(churn)
     
if __name__ == '__main__':
    main()