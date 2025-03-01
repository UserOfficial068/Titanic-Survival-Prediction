import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# import dataset

df_train = pd.read_csv('data/train (1).csv')
df_test = pd.read_csv('data/test.csv')

df_train.head()

# handling missing value



print(df_train.isnull().sum())
print(df_test.isnull().sum())

# filling missing value

df_train["Age"].fillna(df_train["Age"].median(),inplace=True)

df_train['Embarked'].fillna(df_train['Embarked'].mode()[0],inplace=True)

df_train = df_train.drop(['Cabin','Name','Ticket'],errors='ignore')


# converting categorical variable into numerical values


df_train = pd.get_dummies(df_train,columns=['Sex','Embarked'],drop_first=True)

print(df_train)

#spliting the dataset into features and label

x= df_train.drop('Survived',axis=1)
y = df_train['Survived']

# split dataset into training into training and testings sets

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# Train a machine learning model

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train, y_train)

# make validation on the x_test

y_pred = model.predict(x_test)

# evaluate model perfomance

accuracy = accuracy_score(y_test,y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

print(classification_report(y_test,y_pred))







