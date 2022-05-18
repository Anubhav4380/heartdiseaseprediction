#Basic
import numpy as np
import pandas as pd
import os

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Model
import pickle

import streamlit as st 

dirname = os.path.dirname(__file__)

filename = os.path.join(dirname, '..\pickle\knn.pkl')


clf = pickle.load(open(filename, 'rb'))

dataset = pd.read_csv(os.path.join(dirname,'..\data\heart.csv'))

dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal','target'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


def predict_note_authentication(X):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Age
        in: query
        type: number
        required: true
      - name: Sex
        in: query
        type: number
        required: true
      - name: Chest Pain Type
        in: query
        type: number
        required: true
      - name: Resting Blood Pressure
        in: query
        type: number
        required: true
      - name: Serum Cholestoral
        in: query
        type: number
        required: true
      - name: Fasting Blood Sugar
        in: query
        type: number
        required: true
      - name: Resting Electrocardiographic Results
        in: query
        type: number
        required: true
      - name: Maximum Heart Rate Achieved
        in: query
        type: number
        required: true
      - name: Exercise Induced Angina
        in: query
        type: number
        required: true
      - name: ST depression induced by exercise relative to rest 
        in: query
        type: number
        required: true
      - name: Slope of the peak exercise ST segment
        in: query
        type: number
        required: true
      - name: Number of major vessels (0-3) colored by flourosopy
        in: query
        type: number
        required: true
      - name: Thal
        in: query
        type: number
        required: true                                        
    responses:
        200:
            description: The output values
        
    """
   
    prediction=clf.predict(X)
    #print(prediction)

    pred_prob = clf.predict_proba(X)*100
    value_likely = pred_prob[0][1]

    outputstr = None

    #print(prediction[0])
    if(int(prediction[0]) != 0):
      outputstr = 'Presence of Heart Disease'+'('+str(value_likely)+'%)'
    else:
      outputstr='No Presence of Heart Disease'+'('+str(value_likely)+'%)'
    return outputstr


def scale_input(inpu):
    test_dataset = pd.DataFrame(inpu, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    test_dataset = pd.get_dummies(test_dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    test_dataset[columns_to_scale] = standardScaler.transform(test_dataset[columns_to_scale])
    test_dataset = test_dataset.reindex(columns = dataset.columns, fill_value=0)
    X = test_dataset.drop(['target_0', 'target_1'], axis = 1)
    return X

def main():
    st.title("Heart Disease Predict")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Disease Predict ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.number_input("Age",step=1.,format="%.2f") 
    sex= st.number_input("Sex",step=1.,format="%.2f")
    cp= st.number_input("Chest Pain Type",step=1.,format="%.2f")
    trestbps = st.number_input("Resting Blood Pressure",step=1.,format="%.2f")
    chol = st.number_input("Serum Cholestoral",step=1.,format="%.2f")
    fbs = st.number_input("Fasting Blood Sugar",step=1.,format="%.2f")
    restecg = st.number_input("Resting Electrocardiographic Results",step=1.,format="%.2f")
    thalach = st.number_input("Maximum Heart Rate Achieved",step=1.,format="%.2f")
    exang = st.number_input("Exercise Induced Angina",step=1.,format="%.2f")
    oldpeak = st.number_input("ST depression induced by exercise relative to rest ",step=1.,format="%.2f")
    slope = st.number_input("Slope of the peak exercise ST segment",step=1.,format="%.2f")
    ca = st.number_input("Number of major vessels (0-3) colored by flourosopy",step=1.,format="%.2f")
    thal = st.number_input("Thal",step=1.,format="%.2f")
    
    
    result=""
    if st.button("Predict"):
        inpu = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal]]
        X = scale_input(inpu)
        result=predict_note_authentication(X)
    st.success(result)
    result2=""
    if st.button("Predict From file"):
        inpu = []
        flag = False
        with open("..\input\input.txt", "r+") as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                for word in line.split(" "):
                    if not flag:
                        flag = True
                        continue
                    if flag:
                        flag = False
                        inpu.append(float(word))
        inpu = [inpu]
        X = scale_input(inpu)
        result2=predict_note_authentication(X)
    st.success(result2)
    if st.button("About"):
        st.text("Heart Disease Predictor")
        st.text("Final Year Project Submited By:")
        st.text("Muskan Khatwani 1NT18IS202")
        st.text("Anubhav Yadav 1NT18CS017")
        st.text("Neha V M 1NT18CS106")
        st.text("Kinshuk Chaturvedy 1NT18CS076")

if __name__=='__main__':
    main()
