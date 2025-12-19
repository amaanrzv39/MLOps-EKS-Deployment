import numpy as np
import pandas as pd
import pickle
import yaml
from src.logger import logging


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(clf_name: str, params: dict, X_train: np.ndarray, y_train: np.ndarray):
    """Train the Logistic Regression model."""
    try:
        if clf_name == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(**params)
        elif clf_name == 'xgboost':
            from xgboost import XGBClassifier
            clf = XGBClassifier(**params)
        elif clf_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(**params)
        elif clf_name == 'MultinomialNB':
            from sklearn.naive_bayes import MultinomialNB
            clf = MultinomialNB(**params)
        else:
            raise ValueError(f"Unsupported classifier: {clf_name}")
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = load_params('params.yaml')['model_training']['clf']
        params = load_params('params.yaml')['model_training']['hyperparameters'][clf]

        trained_clf = train_model(clf, params, X_train, y_train)

        save_model(trained_clf, 'models/model.pkl')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()