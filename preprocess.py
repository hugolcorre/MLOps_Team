# Function to preprocess the dataset
def preprocess_data(data):
    """Handles missing values and defines features and targets."""
    print("Preprocessing data...")
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data = data.dropna(subset=['Embarked'])
    X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = data['Survived']
    return X, y