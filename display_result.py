# Function to display the evaluation metrics and ROC curve
def display_results(metrics, y_test, y_proba):
    """Displays the metrics and plots the ROC curve."""
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print("\nConfusion Matrix:\n", metrics['Confusion Matrix'])
    print("\nClassification Report:\n", metrics['Classification Report'])
    
    print("Plotting the ROC curve...")
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.show()