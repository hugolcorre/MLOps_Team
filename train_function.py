# Function to train the model
def train_model(pipeline, X_train, y_train):
    """Trains the pipeline on the training data."""
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    return pipeline
