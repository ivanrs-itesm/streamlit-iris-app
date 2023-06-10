import streamlit as st
import pandas as pd
import pickle

st.header("Iris Predictor App 🌸")

st.write("""
## Iris Flower Prediction App with Logistic Regression
### This app will load a pre-trained model to do inference on new instances
""")
         



st.sidebar.header('Choose the Flower Features')


def user_input():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input()

st.subheader('User Input')
st.write(df)

# Load Trained Model
filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))

prediction = model.predict(df)

st.subheader('Prediction: ')
target_names = ["Setosa", "Versicolor", "Virginica"]
st.write(target_names[prediction[0]])
