from my_ml_package.data_ingestion import load_data, preprocess_data
from my_ml_package.model_training import create_pipeline, train_model
from my_ml_package.model_evaluation import evaluate_model, display_results

# Main function
def main():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    # Load and preprocess data
    data = load_data(url)
    X, y = preprocess_data(data)
    
    # Split the dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the pipeline
    pipeline = create_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)
    
    # Evaluate the model
    metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)
    
    # Display results
    display_results(metrics, y_test, y_proba)

if __name__ == "__main__":
    main()
