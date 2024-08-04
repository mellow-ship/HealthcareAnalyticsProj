import joblib
import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from src.features.build_features import create_feature_pipeline

logger = logging.getLogger(__name__)

def load_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['features']['exclude'] = config['features'].get('exclude', [])
    return config

def create_model_pipeline(config):
    numeric_features = config['features']['numeric']
    categorical_features = config['features']['categorical']
    excluded_features = config['features'].get('exclude', [])

    # Remove excluded features
    numeric_features = [f for f in numeric_features if f not in excluded_features]
    categorical_features = [f for f in categorical_features if f not in excluded_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = LGBMRegressor(**config['model']['params'])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

def debug_step(X):
    print("Shape after preprocessing:", X.shape)
    print("First few rows after preprocessing:")
    print(X[:5])
    return X
    
    
def train_model(X_train, y_train, config):
    logger.info("Training model")
    pipeline = create_model_pipeline(config)
    pipeline.fit(X_train, y_train)
    
    # Extract the LightGBM model from the pipeline
    lgbm_model = pipeline.named_steps['regressor']
    
    # Get the actual feature names used by the model
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    return pipeline, lgbm_model, feature_names

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'MSE': mse, 'R2': r2}


def analyze_feature_importance(lgbm_model, config):
    feature_importance = lgbm_model.feature_importances_
    feature_names = config['features']['actual']  # Use the actual features
    
    print(f"Number of feature importance scores: {len(feature_importance)}")
    print(f"Number of feature names: {len(feature_names)}")
    
    # Remove excluded featuresy
    excluded_features = config['features'].get('exclude', [])
    feature_names = [f for f in feature_names if f not in excluded_features]
    
    print(f"Number of feature importance scores: {len(feature_importance)}")
    print(f"Number of feature names: {len(feature_names)}")
    
    # Ensure that the number of features matches the feature importances
    if len(feature_names) != len(feature_importance):
        logger.warning(f"Mismatch between number of features ({len(feature_names)}) and feature importances ({len(feature_importance)})")
        # Use the shorter length to avoid the "All arrays must be of the same length" error
        min_length = min(len(feature_names), len(feature_importance))
        feature_names = feature_names[:min_length]
        feature_importance = feature_importance[:min_length]
    
    # Create a dataframe of feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Create 'figures' directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['feature'], importance_df['importance'])
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png')
    
    # Log the location of the saved plot
    logger.info(f"Feature importance plot saved to: {os.path.abspath('figures/feature_importance.png')}")
    
    # Log top 10 important features
    logger.info("Top 10 important features:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    return importance_df

import os

def save_model(model, file_path):
    logger.info(f"Saving model to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        joblib.dump(model, file_path)
        logger.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    # Load and preprocess data
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    all_features = config['features']['numeric'] + config['features']['categorical']
    excluded_features = config['features'].get('exclude', [])
    features = [f for f in all_features if f not in excluded_features]
    target = config['features']['target']
    
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Train model
    pipeline, lgbm_model, actual_features = train_model(X_train, y_train, config)
    
    # Evaluate model
    metrics = evaluate_model(pipeline, X_test, y_test)
    logger.info(f"Model performance: {metrics}")
    
    # Update config with actual features used
    config['features']['actual'] = actual_features.tolist()
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(lgbm_model, config)
    
    # Save model
    save_model(pipeline, 'models/trained_model.txt')
    
if __name__ == '__main__':
    main()