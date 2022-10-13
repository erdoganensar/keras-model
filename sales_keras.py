import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *


#read_data
training_data_df=pd.read_csv('sales_data_training.csv')

test_data_df=pd.read_csv('sales_data_test.csv')

#0 ve 1 arasına ölçeklendirme
scaler=MinMaxScaler(feature_range=(0,1))

scaled_training=scaler.fit_transform(training_data_df)
scaled_test=scaler.fit_transform(test_data_df)

print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], 
                                                                                                   scaler.min_[8]))


#Yeni dataframe oluşturmak için
scaled_training_df=pd.DataFrame(scaled_training,columns=training_data_df.columns.values)
scaled_test_df=pd.DataFrame(scaled_test,columns=test_data_df.columns.values)

#Ölçeklendirilmiş yeni dataframe kaydetme
scaled_training_df.to_csv("sales_data_training_scaled.csv",index=False)
scaled_test_df.to_csv("sales_data_test_scaled.csv",index=False)

#create model

training_data_df=pd.read_csv("sales_data_training_scaled.csv")

X=training_data_df.drop('total_earnings',axis=1).values
Y=training_data_df[['total_earnings']].values


model=Sequential()
model.add(Dense(32,input_dim=9,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')

#Training model

model.fit(
    X,
    Y,
    epochs=50,
    verbose=2
    )

# Değerlendirme

test_data_df=pd.read_csv("sales_data_test_scaled.csv")

X_test=test_data_df.drop('total_earnings',axis=1).values
Y_test=test_data_df[['total_earnings']].values

test_error_rate=model.evaluate(X_test,Y_test,verbose=0)
print("The mean squared error for the test data set is : {}".format(test_error_rate))

#Prediction
X=pd.read_csv('proposed_new_product.csv')

prediction=model.predict(X)

prediction=prediction[0][0]

prediction=prediction + 0.1159
prediction=prediction / 0.0000036968

print("Earning Predictions  for Proposed Product -${}".format(prediction))

#save model
model.save('training_model.h5py')
print("Model save to disk")
