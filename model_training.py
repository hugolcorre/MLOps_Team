from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def create_pipeline():
    """Creates a pipeline for preprocessing and logistic regression."""
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
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    return pipeline

def train_model(pipeline, X_train, y_train):
    """Trains the pipeline on the training data."""
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    return pipeline
