# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Loading and Preprocessing: The program begins by loading the employee_churn_dataset.csv file into a pandas DataFrame. It then performs data cleaning by dropping the 'ID' column if present and creates a new target variable 'Attrition' based on 'Tenure' and 'Salary' conditions. Categorical features like 'Gender', 'Education Level', 'Marital Status', 'Job Role', 'Department', and 'Work Location' are then numerically encoded using LabelEncoder.

2.Data Splitting: The preprocessed data is divided into features (X) and the target variable (y). The dataset is then split into training and testing sets (X_train, X_test, y_train, y_test) using train_test_split with a 80/20 ratio and a fixed random_state for reproducibility.

3.Model Training: A Decision Tree Classifier (DecisionTreeClassifier) is initialized with a maximum depth of 3 to prevent overfitting. This model is then trained on the training data (X_train, y_train) using the fit method.

4.Model Evaluation: The trained Decision Tree model is used to make predictions on the unseen test data (X_test). The accuracy of these predictions is then calculated and printed using metrics.accuracy_score, comparing the predicted values (y_pred) with the actual test labels (y_test).

5.Feature Importance and Visualization: The program calculates and displays the top 3 most important features identified by the Decision Tree model, providing insights into which factors contribute most to employee attrition. Finally, it visualizes the trained decision tree structure using matplotlib.pyplot and plot_tree, offering a graphical representation of the decision-making process.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIGNESH J
RegisterNumber:25014705  
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data
data = pd.read_csv("employee_churn_dataset.csv")
print(data.head())

# Drop ID column
if 'ID' in data.columns:
    data = data.drop(['ID'], axis=1)

# Create target variable
data['Attrition'] = ((data['Tenure'] < 2) & (data['Salary'] < data['Salary'].median())).astype(int)
print(f"Created Attrition: {data['Attrition'].value_counts().to_dict()}")

# Encode text columns
le = LabelEncoder()
for col in ['Gender', 'Education Level', 'Marital Status', 'Job Role', 'Department', 'Work Location']:
    data[col] = le.fit_transform(data[col])

# Split data
X = data[['Age', 'Gender', 'Education Level', 'Marital Status', 
          'Tenure', 'Job Role', 'Department', 'Salary', 'Work Location']]
y = data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train model
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

# Test model
y_pred = dt.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.3f}")

# Show top features
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTop 3 Features:")
print(importance.head(3))

# Show tree
plt.figure(figsize=(10, 6))
plot_tree(dt, feature_names=X.columns.tolist(), 
          class_names=['Stayed', 'Left'], filled=True)
plt.show()
```

## Output:
![decision tree classifier model](sam.png)
<img width="1318" height="766" alt="Screenshot 2025-10-05 213818" src="https://github.com/user-attachments/assets/489a838e-ce3a-41a7-a7ac-63d192d4aa40" />
<img width="1394" height="776" alt="Screenshot 2025-10-05 213834" src="https://github.com/user-attachments/assets/23e00cd8-9e9e-4ab2-9b80-4d48ffc2dc3f" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
