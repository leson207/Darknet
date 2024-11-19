from src.entity.configuration import ConfigManager
from src.component.Predictor import Predictor

from PIL import Image

if __name__=="__main__":
    config_manager = ConfigManager()

    predictor_config=config_manager.get_predictor_config()
    predictor=Predictor(predictor_config)
    img=Image.open('input.png')
    predictor.inference(img).show()