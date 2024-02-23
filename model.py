#import modules
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import joblib
import seaborn as sb
df=pd.read_csv('Stress Data_C.csv')

#select dependent and independent variables
X = df[['snoring_rate', 'body_temperature','blood_oxygen','respiration_rate', 'sleeping_hours', 'heart_rate','Headache','Working_hours']]
y = df['stress_level']

#split the dataset into test and train data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#feature scaling
'''sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)'''

#instantiate model
random_forest_classifier=RandomForestClassifier(n_estimators=100, random_state=42)

#fit the model
random_forest_classifier.fit(X_train,y_train)
random_forest_classifier.predict(X_test)
#pickle file of our model
#pickle.dump(random_forest_classifier,open("model.pkl","wb"))
#joblib.dump(random_forest_classifier, "model.joblib")
pickle.dump(random_forest_classifier,open('model.pklz','wb'))


