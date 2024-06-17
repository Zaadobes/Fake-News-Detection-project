#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import string


#Reading dataset and putting them in varaiable
real01= pd.read_csv("file path")
fake01= pd.read_csv("file path")

#Converting dataset from parquet  to csv
real02= pd.read_parquet('file path')
real02.to_csv('file path')

#Viewing dataset (the first 5 rows of dataset and last 5 respectively)
real01.head()


fake01.tail() 

#Creating a new column class in dataset with the value 1 or all rows
real01["class"]= 1 
real02["class"]=1
fake01["class"]= 0

#Shape function shows the size of the dataset in rows and columns
real01.shape , fake01.shape , real02.shape 

#Plotting realo1 against fake01 in a graph
real_lengths = real01['text'].apply(len)
fake_lengths = fake01['text'].apply(len)
plt.hist(real_lengths, bins=50, alpha=0.5, label='Real')
plt.hist(fake_lengths, bins=50, alpha=0.5, label='Fake')
plt.title('Article Lengths')
plt.xlabel('Length')
plt.ylabel('Count')
plt.legend()
plt.show()


#Creating a new column class in dataset with the value 1 or 0 for all rows to classify truor false dataset
real01["class"]= 1 
real02["class"]=1
fake01["class"]= 0

#Creating a duplicate coluum to rename column to match the other dataset
real02["text"]=real02["content"]
real02.head()

#Merging datasets
merged_data=pd.concat([fake01,real01,real02],axis=0)
merged_data.head()

#Dropping columns not needed in training model
data = merged_data.drop(['content','category','author',
                       'subject','date','published_date','page_url','title'],axis=1)
data.head()

#Checking for null values in dataset
data.isnull().sum()
data.isnull().sum().sum() #total null values present

#Removing all rows coontaining null values
data1 = data.dropna()
data = data1

data.isnull().sum().sum()

#Preprocessing dataset
def preprocess(text):
  text =text.lower()
  text =re.sub('\[.*?]', '', text)
  text =re.sub("\\W", " ",  text)
  text =re.sub('htps?://\S+|ww.\.\S+', '', text)

  text =re.sub('<.*?>+', '',  text)
  text =re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text =re.sub('\n', '', text)
  text =re.sub('\w*\d\w*', '',  text)
  return text

data ['text'] = data['text'].apply(preprocess)
#Assigning x_matrix and y value of data
x = data['text']
y = data['class']

#Splitting dataset
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)  # 60% train, 40% temp

# Second split: validation and test
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)  # 50% of temp each for validation and test

# # Output the results
print("x_train:", x_train)
print("x_val:", x_val)
print("x_test:", x_test)
print("y_train:", y_train)
print("y_val:", y_val)
print("y_test:", y_test)

#Vectorizing text
from sklearn.feature_extraction.text import TfidfVectorizer
vectorize= TfidfVectorizer()
m_train= vectorize.fit_transform(x_train)
m_valid = vectorize.transform(x_val)
m_test = vectorize.transform(x_test)
# print(m_test)


#MODELS
#KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=1)
KNN.fit(m_train,y_train)
y_pred = KNN.predict(m_test)
KNN.score(m_test ,y_test)
print(classification_report(y_test,y_pred))

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
LoG = LogisticRegression()
LoG.fit(m_train,y_train)
y_pred = LoG.predict(m_test)
LoG.score(m_test ,y_test)
print(classification_report(y_test,y_pred))

#SVM MODEL
from sklearn.svm import SVC
SVM = SVC()
SVM = SVM.fit(m_train,y_train)
y_pred = SVM.predict(m_test)
SVM.score(m_test ,y_test)
print(classification_report(y_test,y_pred))

#LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
LiG = LinearRegression()
LiG = LiG.fit(m_train,y_train)
y_pred = LiG.predict(m_test)
LiG.score(m_test ,y_test)
print(classification_report(y_test,y_pred))

#OUTPUT
def output(n):
  if n == 0:
   return "Fake News"
  elif n == 1:
   return "Not A Fake News"

#Function to process input
def news_detection(news):
  testing_news = {"text":[news]}
  news_test = pd.DataFrame(testing_news)
  news_test["text"] = news_test["text"].apply(preprocess)
  news_m_test = vectorization.transform(news_test)
  KNN_pred = KNN.predict(new_m_test)
  LoG_pred = LoG.predict(new_m_test)
  SVM_pred = SVM.predict(new_m_test)
  LiG_pred = LiG.predict(new_m_test)
  return print("\n\nKNN Prediction: {} \nLoG PredictionLoG Prediction: {} \nSVM Prediction: {} \nLiG Prediction: {}"
    output(KNN_pred[0]),output(LoG_pred[0]),output(SVM_pred[0]),output(LiG_pred[0]))

#taking input and calling detectin function
news = str(input())
news_detection(news)



