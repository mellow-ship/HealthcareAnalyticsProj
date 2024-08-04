import logging
from src.data.make_dataset import main as make_dataset
from src.models.train_model import main as train_model

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Making dataset")
    make_dataset()
    
    logger.info("Training model")
    train_model()
    
    logger.info("Process completed successfully")

if __name__ == '__main__':
    main()