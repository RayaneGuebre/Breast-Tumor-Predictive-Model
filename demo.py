import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




def algorithm (array):
    
    df = pd.read_csv("Breast_cancer_dataset.csv")
    print(array)
    features_list = df.columns.drop("id").drop("diagnosis").drop("Unnamed: 32")
    df = df.drop(["id"], axis=1)
    df = df.drop(["Unnamed: 32"], axis=1)
    df["diagnosis"] = df["diagnosis"].apply(lambda x:1 if x == "M" else 0)
    x_train, x_test, y_train, y_test = train_test_split(df.drop("diagnosis", axis=1), df["diagnosis"])
    LogReg = LogisticRegression(solver="lbfgs", max_iter=10000)
    LogReg.fit(x_train, y_train)
    features = x_train.columns
    prediction_data = pd.DataFrame([array], columns=features)

    prediction = LogReg.predict(prediction_data)[0]
    if prediction == 1:
        return "The tumor is malignant"
    else:
        return "The tumor is benign"



with st.container(vertical_alignment="center"):
    st.title("Breast tumor Predictor")
    st.write("Here are some arrays to try the model")
    st.write("Benign: 13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259")
    st.write("Malign: 15.46,11.89,102.5,736.9,0.1257,0.1555,0.2032,0.1097,0.1966,0.07069,0.4209,0.6583,2.805,44.64,0.005393,0.02321,0.04303,0.0132,0.01792,0.004168,18.79,17.04,125,1102,0.1531,0.3583,0.583,0.1827,0.3216,0.101")
    st.write("here what each value means, for the expmple i used the malign array.")
    data = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",],[15.46,11.89,102.5,736.9,0.1257,0.1555,0.2032,0.1097,0.1966,0.07069,0.4209,0.6583,2.805,44.64,0.005393,0.02321,0.04303,0.0132,0.01792,0.004168,18.79,17.04,125,1102,0.1531,0.3583,0.583,0.1827,0.3216,0.101]
    st.dataframe(data)
    input_array = st.text_input("Input Array")
    input_array = input_array.split(",")
    if st.button("Predict"):
        st.title(algorithm(input_array))

