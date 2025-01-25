#Test script
from .data_ingestion import load_data, preprocess_data
from .model_training import create_pipeline, train_model
from .model_evaluation import evaluate_model, display_results
from .inference import make_predictions
