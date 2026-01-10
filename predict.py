import mlflow
import joblib
from pathlib import Path

def getModels():

    MODELS_DIR = Path(__file__).resolve().parent / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.pkl"

    if model_path.exists(): 
        model = joblib.load(model_path)

    else:
        mlflow.set_tracking_uri('http://127.0.0.1:5000/')
        registered = mlflow.search_registered_models(filter_string="name = 'bestXGB'")
        latest_version = max([version.version for version in registered[0].latest_versions])

        model = mlflow.sklearn.load_model(f'models:/bestXGB/{latest_version}')

        joblib.dump(model, model_path)

    features = model.feature_names_in_

    return model, features