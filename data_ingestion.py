import pandas as pd

def load_data(url):
    """Loads the dataset from a given URL."""
    print("Loading dataset...")
    return pd.read_csv(url)

def preprocess_data(data):
    """Handles missing values and defines features and targets."""
    print("Preprocessing data...")
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data = data.dropna(subset=['Embarked'])
    X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = data['Survived']
    return X, y
