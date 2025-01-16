#This is the code the prof gave us : 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Step 1: Load the Titanic dataset
print("Loading the Titanic dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)
# Step 2: Inspect the dataset
print("Inspecting the dataset...")
print("Dataset Head:\n", data.head())
print("Dataset Info:\n")
print(data.info())
# Step 3: Handle missing values
print("Handling missing values...")
data['Age'] = data['Age'].fillna(data['Age'].median())
data = data.dropna(subset=['Embarked'])
print("Missing values handled.")
# Step 4: Define features and target
print("Defining features and target...")
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']
print("Features and target defined.")
# Step 5: Preprocessing
print("Setting up preprocessing pipelines...")
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
# Step 6: Train-test split
print("Splitting the dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split completed.")

# Step 7: Build the pipeline
print("Building the pipeline...")
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])
# Step 8: Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Model training completed.")
# Step 9: Make predictions
print("Making predictions...")
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
print("Predictions completed.")
# Step 10: Evaluate the model
print("Evaluating the model...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)
# ROC Curve
print("Plotting the ROC curve...")
roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
roc_display.figure_.suptitle("ROC Curve")
# Show metrics and visualizations
import matplotlib.pyplot as plt
plt.show()

print("Pipeline execution completed.")

