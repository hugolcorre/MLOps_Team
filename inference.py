def make_predictions(pipeline, X_new):
    """Uses the trained pipeline to make predictions on new data."""
    return pipeline.predict(X_new), pipeline.predict_proba(X_new)[:, 1]
