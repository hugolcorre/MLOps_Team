from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, RocCurveDisplay
)
import matplotlib.pyplot as plt

def evaluate_model(pipeline, X_test, y_test):
    """Evaluates the model and returns metrics and predictions."""
    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, output_dict=False)
    }
    return metrics, y_pred, y_proba

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
