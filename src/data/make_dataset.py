import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


logger = logging.getLogger(__name__)

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def process_data(df, config):
    logger.info("Processing data")
    
    # Handle 'Age' variable
    age_categories = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    age_encoder = OrdinalEncoder(categories=[age_categories])
    df['Age_Ordinal'] = age_encoder.fit_transform(df[['Age']])
    
    # Handle 'Stay' variable
    stay_categories = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', 
                       '61-70', '71-80', '81-90', '91-100', 'More than 100 Days']
    stay_encoder = OrdinalEncoder(categories=[stay_categories])
    df['Stay'] = stay_encoder.fit_transform(df[['Stay']])
    
    # Handle missing values in numeric columns
    for col in config['features']['numeric']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
    
    # Handle missing values in categorical columns
    for col in config['features']['categorical']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Select relevant features
    features = config['features']['numeric'] + config['features']['categorical']
    target = config['features']['target']
    df = df[features + [target]]
    
    return df

def split_data(df, config):
    logger.info("Splitting data into train and test sets")
    return train_test_split(df, test_size=config['training']['test_size'], 
                            random_state=config['training']['random_state'])

def main():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    raw_data = load_data(config['data']['raw_data_path'])
    processed_data = process_data(raw_data, config)
    
    train_data, test_data = split_data(processed_data, config)
    
    train_data.to_csv('data/processed/train_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)

if __name__ == '__main__':
    main()