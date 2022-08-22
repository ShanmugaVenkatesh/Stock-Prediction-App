import pandas_datareader as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st


start = '2002-01-01'
end = '2022-07-31'


st.title("Stock Trend Prediction ")

user_input = st.text_input("Enter Stock Ticker ", 'SBIN.NS')
dataframe=data.DataReader(user_input, 'yahoo', start, end)


#Describing the Data
st.subheader("Data from 2002 - 2022")
st.write(dataframe.describe())


#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(19,10))
plt.plot(dataframe.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = dataframe.Close.rolling(100).mean()
fig = plt.figure(figsize=(19,10))
plt.plot(ma100,'r')
plt.plot(dataframe.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = dataframe.Close.rolling(200).mean()
ma100 = dataframe.Close.rolling(100).mean()
fig = plt.figure(figsize=(19,10))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(dataframe.Close)
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(dataframe['Close'][0:int(len(dataframe)*0.7)])
data_testing = pd.DataFrame(dataframe['Close'][int(len(dataframe)*0.7) : int(len(dataframe))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



model = load_model("keras_model.h5")

#Testing Part

past_100_days = data_training.tail()
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])    

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

#Final Graph 

st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(20,10))
plt.plot(y_test ,"b" ,label = "Original Price" )
plt.plot(y_predicted, 'r', label = "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)