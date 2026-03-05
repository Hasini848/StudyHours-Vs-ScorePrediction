import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   

st.title("Exam Score based on Hours of Study")
st.write("Upload your dataset")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)
    
    st.write("Your dataset is:")
    st.dataframe(data.head())
    
    if 'Hours' in data.columns and 'Scores' in data.columns:
        
        X = data[['Hours']]
        Y = data['Scores']
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, Y_train)
        
        st.subheader("Input Hours of Study")
        hours = st.number_input(
            "Enter hours of study",
            min_value=0.0,
            max_value=24.0,
            value=5.0
        )
        
        if st.button("Predict"):
            input_data = np.array([[hours]])
            prediction = model.predict(input_data)
            
            st.success(f"Predicted Score: {prediction[0]:.2f}")
            
            # Graph for predicted point only
            fig, ax = plt.subplots()
            ax.scatter(hours, prediction[0])
            ax.set_xlabel("Hours of Study")
            ax.set_ylabel("Predicted Score")
            ax.set_title("Predicted Score for Given Hours")
            
            st.pyplot(fig)
            
    else:
        st.error("Dataset must contain 'Hours' and 'Scores' columns.")