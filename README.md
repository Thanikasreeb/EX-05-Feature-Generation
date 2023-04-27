# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM

### STEP 1

Read the given Data

### STEP 2

Clean the Data Set using Data Cleaning Process

### STEP 3

Apply Feature Generation techniques to all the feature of the data set

### STEP 4

Save the data to the file

# CODE
```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
```
Titanic.csv :

import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

# removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

# data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

# feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT

# Data.csv:

## Initial Dataset:
![image](https://user-images.githubusercontent.com/119557910/234955771-9d2e5276-2a26-467f-8bf9-32303e60a28e.png)

## Binary Encoding:
![image](https://user-images.githubusercontent.com/119557910/234955849-369a45ab-371c-46d6-b79c-cd7ff6f3e434.png)

## Encoded Dataset:
![image](https://user-images.githubusercontent.com/119557910/234956097-7e80c643-99b1-4eb5-ad69-eff8d88a10e2.png)

## Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/119557910/234956173-42966b0c-c2f6-4c8b-9218-96bfcc07f37b.png)

## Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/119557910/234956258-fef1cd3c-1417-491f-9370-6c06c04b8a81.png)

## Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/119557910/234956342-4617242a-0df6-42c8-ba27-9f809304656d.png)

## Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/119557910/234956404-c1e35c01-9ec9-4cfe-bb9c-92a0adccef3a.png)

## Encoding.csv :
 
## Initial Dataset:
![image](https://user-images.githubusercontent.com/119557910/234956592-e1f1ff6b-1fbb-4883-a0b9-5f78d808dfac.png)

## Binary Encoding:
![image](https://user-images.githubusercontent.com/119557910/234956661-d2c6c229-7f07-44c5-86a4-7245af0ebca4.png)

## Encoded Dataset:
![image](https://user-images.githubusercontent.com/119557910/234956727-423160fc-feb0-4f59-8fb4-f2f683602a9d.png)

![image](https://user-images.githubusercontent.com/119557910/234956754-d91a1b66-f17e-4df1-88bb-1931ecb45d81.png)

## Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/119557910/234956867-835ef37c-3653-43fe-83bb-e7a113ff914c.png)

## Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/119557910/234956952-bba7fc18-a8dd-4e45-8c1b-5206b5a21cd8.png)

## Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/119557910/234957010-9583c829-d3d0-4637-be5a-cc94b4b79be1.png)

## Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/119557910/234957102-eaf34984-f4bc-4b39-960c-45239a5339c7.png)

## Titanic.csv : Initial Dataset:
![image](https://user-images.githubusercontent.com/119557910/234957174-f4361341-ada7-4064-8dbc-0dbd112f4e46.png)

## Data cleaning before encoding:
![image](https://user-images.githubusercontent.com/119557910/234957283-83803ce6-9940-4b22-afa5-2a92d6c04fc3.png)
![image](https://user-images.githubusercontent.com/119557910/234957321-f2ed9cef-df7b-4d62-ba8f-6c1ef371ca48.png)
![image](https://user-images.githubusercontent.com/119557910/234957348-9e683f91-24f3-4584-b165-3308c52845b1.png)

## Cleaned Dataset:
![image](https://user-images.githubusercontent.com/119557910/234957424-fcb6d23a-7c98-4f6e-9c7e-451e67652b28.png)

## Binary Encoding:
![image](https://user-images.githubusercontent.com/119557910/234957550-ebf9e927-0005-4f3f-812c-6ad4d45e8ca0.png)

## Encoded Dataset:
![image](https://user-images.githubusercontent.com/119557910/234957626-9c579148-a2b4-4ab4-ae31-4290ff7ff85f.png)

## Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/119557910/234957700-bd263dcf-00d9-45de-88bc-8fae4dceb8a9.png)

## Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/119557910/234957790-e80a1301-6a80-4acb-86bf-5b2e9d09fdec.png)

## Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/119557910/234957844-fb3b008e-1e85-45da-bd34-9d031097ff8c.png)

## Result:

Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.






