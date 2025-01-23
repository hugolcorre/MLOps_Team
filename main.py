# Main function to orchestrate the workflow
def main():
    """Main function to execute the machine learning pipeline."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    # Load and preprocess data
    data = load_data(url)
    X, y = preprocess_data(data)
    
    # Split the dataset
    print("Splitting the dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the pipeline
    pipeline = create_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)
    
    # Evaluate the model
    metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)
    
    # Display results
    display_results(metrics, y_test, y_proba)