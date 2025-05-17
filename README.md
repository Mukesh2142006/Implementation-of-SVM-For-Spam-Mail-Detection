# EX-11:Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and extract the text messages (X) and their labels (Y).
2. Split the data into training and testing sets.
3. Convert text to numeric features using TF-IDF vectorization.
4. Train a Support Vector Machine (SVM) classifier on the training data.
5. Predict the labels for the test set using the trained model.
6. Evaluate the model using accuracy or other metrics.

## Program:

```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MUKESH
RegisterNumber:  212223230128
import pandas as pd
df=pd.read_csv('spam (1).csv',encoding='Windows-1252')
df.head()
df.tail()
df.info()
df.isnull().sum()
X=df['v2']
Y=df['v1']
X.shape
Y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
x_train=tf.fit_transform(x_train)
x_test=tf.transform(x_test)
x_train
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
*/
```

## Output:
![image](https://github.com/user-attachments/assets/04c6aced-8487-4642-81c9-d72f2fa6d03c)<br>
![image](https://github.com/user-attachments/assets/39bd1ef8-fec1-4962-8252-e7c774be148f)<br>
![image](https://github.com/user-attachments/assets/bf39c4b5-c1f6-4815-8a05-a40efed7ce78)<br>
![image](https://github.com/user-attachments/assets/ed284a8f-2fa3-4e66-9f7b-b57183e09566)<br>
![image](https://github.com/user-attachments/assets/62cef133-1dfe-438c-9082-b816ae35471d)<br>
![image](https://github.com/user-attachments/assets/25b16ba2-1dbb-403b-8407-8d5251de23d0)<br>
![image](https://github.com/user-attachments/assets/b20ca7eb-d4ce-4f99-95f6-436054bc243e)<br>
![image](https://github.com/user-attachments/assets/7ee04bbb-a890-4138-9e8c-84986b5ca2fb)<br>
![image](https://github.com/user-attachments/assets/fce9b599-65b7-43dd-ab79-2eee6ecbde08)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
