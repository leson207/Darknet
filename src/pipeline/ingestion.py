from src.entity.configuration import ConfigManager
from src.component.DataIngestor import DataIngestor

if __name__=="__main__":
    config_manager = ConfigManager()

    data_ingestor_config=config_manager.get_data_ingestor_config()
    data_ingestor=DataIngestor(data_ingestor_config)
    data_ingestor.create_dataset()
    data_ingestor.create_label_mapping()
    # data_ingestor.sync_to_s3()
    # data_ingestor.sync_from_s3()