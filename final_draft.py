#!/usr/bin/env python
# coding: utf-8

# In[10]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import*
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



df=pd.read_csv('train.csv')

pd.crosstab(df['Sex'],df['Age'])

df["Age"].fillna(df["Age"].median(),inplace=True)
df["Fare"].fillna(df["Fare"].median(),inplace=True)


df["Embarked"].fillna("S", inplace=True)

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "Missing title"
   

titles=set([x for x in df.Name.map(lambda x : get_title(x))])

def shorter_titles(x):
    titles=x['Titles']
    if titles in ['Capt','Major','Col']:
        return "Officer"
    elif titles in ["Don",'Jonkheer','the Countess','Master']:
        return "Royalty"
    elif titles in ['Mrs','Lady','Sir','Miss','Mr', 'Ms','Mlle','Mme']:
        return "Common Citizen"
    else:
        return titles
 
df["Titles"]=df.Name.map(lambda x : get_title(x))
df["Titles"]=df.apply(shorter_titles,axis=1)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna("S", inplace=True)
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin", axis=1, inplace=True)
df.Sex.replace(('male',"female"),(0,1),inplace=True)
df.Embarked.replace(('S',"C","Q"),(0,1,2),inplace=True)
df.Titles.replace(('Officer',"Royalty","Dr","Rev","Common Citizen",),(0,1,2,3,4),inplace=True)

oneh = OneHotEncoder(handle_unknown="ignore")
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked","Titles"]
oneh.fit(df[features])
X = pd.get_dummies(df[features])# independant features 
y=df['Survived'] # dependant variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

randomforest=RandomForestClassifier()
randomforest.fit(X_train,y_train)


X_test = oneh.transform(df[features])
X_test = pd.get_dummies(df[features])

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)


def prediction_model(pclass,sex, age, sibSp, parch, fare, embarked, title):
    x=[[pclass,sex, age, sibSp, parch, fare, embarked, title]]
    randomforest=pickle.load(open('titanic_model.sav','rb'))
    predictions=randomforest.predict(x)

