import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data['model']
le_count = data['Le_country']
le_education = data['le_education']
education_level = data['education']
countries = data['countries']


def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write('''### We need some informations to predict the salary''')

    country = st.selectbox("Country", countries)
    education = st.selectbox('Education Level', education_level)
    experience = st.slider("Years of Experience", 0, 50, 2)
    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_count.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)
        salary = regressor.predict(X)
        st.subheader(f'The estimated salary is ${salary[0]:.2f}')
