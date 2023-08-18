#!/usr/bin/env python
# coding: utf-8
#Load required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#Load the data sets
class_data=pd.read_csv('C:/Users/Satyaanalamadhusamhi/Desktop/class.csv')
zoo_data=pd.read_csv('C:/Users/Satyaanalamadhusamhi/Desktop/zoo.csv')
class_data
zoo_data.head()
class_data.shape
zoo_data.shape
#Merge the 2 data sets
data=pd.merge(class_data,zoo_data,left_on='Class_Number',right_on='class_type')
data.head()
#Check for mismatched rows
mis_data=data[data['Class_Number']!=data['class_type']]
mis_data.shape
#Identifying similar columns
data['class_type'].value_counts(),data['Class_Number'].value_counts(),data['Class_Type'].value_counts()
#Two of the columns can be removed
#Number_Of_Animal_Species_In_Class : can be derived/changed during handling imbalance
#Animal_Names : Not necessary
data=data.drop(['class_type','Class_Number','Number_Of_Animal_Species_In_Class','Animal_Names'],axis=1)
data.head()
#Visualisation
plt.figure(figsize=(8,6))
plt.grid(True)
ax=sns.countplot(x='Class_Type',data=data,palette='Spectral_r')
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()),(p.get_x()+0.3,p.get_height()+0.5))

#Visualisation
plt.figure(figsize=(8,6))
plt.grid(True)
cols=['Class_Type','hair']
data_hair=data[cols]
ax=sns.countplot(x='Class_Type',data=data_hair,palette='Spectral_r')
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()),(p.get_x()+0.3,p.get_height()+0.5))sns.countplot(x='hair',data=data)

sns.barplot(x='Class_Type',y='eggs',data=data)

#Instead of visualizing all the 16 features to derive correlation among attributes,
#We can use correlation matrix
features=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
subset_data=data[features]
corr_mat=subset_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_mat,annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
#Pie chart to show population ratio of all the 7 classees
from matplotlib import pyplot as plt
import numpy as np
 fig = plt.figure(figsize =(8, 6))
classes=class_data['Class_Type']
species=class_data['Number_Of_Animal_Species_In_Class']
plt.pie(species, labels = classes)
plt.show()
#We can see that the target variable i.e, Class_Type is unbalanced but we have to measure the imbalance
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
def examine_dataset(df,target_col):
    class_counts=df[target_col].value_counts()
    print("Class Distribution:")
    print(class_counts)
    imbalance_ratio=class_counts.iloc[0]/class_counts.iloc[1]
    print("Imbalance Ratio : ",imbalance_ratio)
    return imbalance_ratio
def handle_imbalanced_data(df,target_col):
    imbalance=examine_dataset(df,target_col)
    if imbalance <= 2:
        print("No imbalance found in the dataset.")
        return
    print("Select an option to handle the imbalanced dataset:")
    print("1. Random Oversampling")
    print("2. Proceed without handling")
    choice=input("Enter your choice(1/2):")
    
    #Separate the features and target variable
    x=df.drop(target_col,axis=1)
    y=df[target_col]
    
    if choice=='1':
        over=RandomOverSampler()
        x_resampled,y_resampled=over.fit_resample(x,y)
    elif choice=='2':
        print("Proceeding without handling the imbalance.")
        return df
    else:
        print("Invalid choice. Proceeding without handling the imbalance.")
    
    #creating new balanced df
    balanced_df=pd.concat([x_resampled,y_resampled],axis=1)
    return balanced_df

data=handle_imbalanced_data(data,'Class_Type')
#Check if the data is balanced
handle_imbalanced_data(data,'Class_Type')
data
#Modelling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#Select features and target variable
features=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
target='Class_Type'
#Spplit the data into training and testing data
X=data[features]
Y=data[target]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#Train the model
rfc=RandomForestClassifier()
rfc.fit(X_train,Y_train)
#Perform classification using test data
Y_pred=rfc.predict(X_test)
print("Comparision of original test data and predicted values")
df=pd.DataFrame({'Actual_values':Y_test,'Predicted_values':Y_pred})
print(df)
print('\n\n')
#Evaluate the performance
report=classification_report(Y_test,Y_pred)
print("Classification Report")
print(report)

#Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
confusion_mat=confusion_matrix(Y_test,Y_pred)
confusion_mat
# In[35]:
#Test
inputs=[]
for i in features:
    user_input=input(f"Enter value for {i}:")
    inputs.append(int(user_input))
user_data=pd.DataFrame([inputs],columns=features)
prediction=rfc.predict(user_data)
                       
print(f"\nPredicted class: {prediction[0]}")













