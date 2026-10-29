# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1:
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2:
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3:
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4:
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.


## PROGRAM

### Name: SUJIN M L

### Register Number: 212225040435


```
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

df=pd.read_csv("customers.csv")
df.shape
df

df=df.drop(columns="ID")
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)

cat_cols=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Var_1"]
for col in cat_cols:
    df[col]=LabelEncoder().fit_transform(df[col])
df

encoder = LabelEncoder()
x=df.drop(columns="Segmentation")
y = encoder.fit_transform(df["Segmentation"])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler1=StandardScaler()
scaler2=StandardScaler()
x_train=scaler1.fit_transform(x_train)
x_test=scaler2.fit_transform(x_test)

x_train=torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)
y_train=torch.FloatTensor(y_train)
y_test=torch.FloatTensor(y_test)

train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=16)

class classification_model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,8)
        self.l4=nn.Linear(8,4)
        
    def forward(self,x):
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=torch.relu(self.l3(x))
        x=self.l4(x)
        return x

input_size=x_train.shape[1]
model=classification_model(input_size)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=200
for i in range(epochs):
    for a,b in train_loader:
        optimizer.zero_grad()
        pred=model(a)
        loss=criterion(pred,b.long())
        loss.backward()
        optimizer.step()

    if i%20==0:
        print(f"Epoch {i}/{epochs}, Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():
    outputs = model(x_test.float())
    pred = torch.argmax(outputs, dim=1)
    loss = criterion(outputs, y_test.long().view(-1))
    print("Test Loss:", loss.item())

accuracy = accuracy_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred, target_names=['A','B','C','D'])
conf_matrix = confusion_matrix(y_test,pred)
print("DEVELOPED BY SUJIN M L ")
print("REGISTER NUMBER : 212225040435")
print("Accuracy Score :",accuracy)
print("Classification Report :\n",class_report)
print("Confusion Matrix :\n",conf_matrix)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```

### OUTPUT

## DATAFSET
<img width="1327" height="463" alt="image" src="https://github.com/user-attachments/assets/3892f1ea-65f7-4a1a-a730-39974da96dea" />

<img width="1206" height="450" alt="image" src="https://github.com/user-attachments/assets/4da76e05-5f4c-4551-8115-03f37aec720c" />

## EPOCH AND LOSS
<img width="217" height="187" alt="image" src="https://github.com/user-attachments/assets/693cb870-5e57-4527-acb1-fd53346f830e" />

## TEST LOSS
<img width="435" height="60" alt="image" src="https://github.com/user-attachments/assets/471d79be-fca2-4fed-83bb-d9543424853d" />

## PERFORMANCE METRICES
<img width="527" height="752" alt="image" src="https://github.com/user-attachments/assets/7bdb88d2-8619-41a3-9233-f65272a98a29" />


### NEW SAMPLE DATA PREDICTION
<img width="583" height="70" alt="image" src="https://github.com/user-attachments/assets/f234c416-0fd7-4c42-806b-e8552521d7b2" />


## RESULT
Thus the python program to develop a neural network classification model is executed successfully
