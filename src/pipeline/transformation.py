from src.entity.configuration import ConfigManager
from src.component.Preprocessor import Preprocessor

if __name__=="__main__":
    config_manager = ConfigManager()

    preprocessor_config=config_manager.get_preprocessor_config()
    preprocessor=Preprocessor(preprocessor_config)
    preprocessor.save()