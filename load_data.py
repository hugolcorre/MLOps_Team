# Function to load the dataset
def load_data(url):
    """Loads the dataset from a given URL."""
    print("Loading dataset...")
    return pd.read_csv(url)