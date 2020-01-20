import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model

maindf = pd.read_csv("H:\\python jupyter\\aa\\t.csv")
maindf

df = pd.read_csv("H:\\python jupyter\\aa\\t.csv")
df

inputs = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1, inplace=True)

df

from sklearn.preprocessing import LabelEncoder

l_sex = LabelEncoder()

df['sex_l'] = l_sex.fit_transform(df['Sex'])

df

inputs = df.drop(['Sex'],axis=1, inplace=True)

df


import math

age = math.floor(df.Age.median())

age

df.Age = df.Age.fillna(age)

df

from word2number import w2n

df.Pclass = df.Pclass.fillna("zero")

df.Pclass

df.Pclass = df.Pclass.apply(w2n.word_to_num)

df.Pclass

df.Survived, df.Pclass,df.Age

df.shape



from sklearn import tree

trainmodel = tree.DecisionTreeClassifier()

target = df['Survived']
target

df.drop('Survived',axis='columns')

target


predict = df.drop('Survived',axis='columns')

predict

trainmodel.fit(predict,target)

trainmodel.score(predict,target)

trainmodel.predict([[3,22,7.2500,1,]])

trainmodel.predict([[3,22,7.2500,1,]])

trainmodel.predict([[3,26,7.9250,0,]])


