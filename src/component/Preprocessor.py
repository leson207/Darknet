import cv2
import joblib
import albumentations as A

from albumentations.pytorch import ToTensorV2

from src.entity.entity import PreProcessorConfig
from src.logger import logger

class Preprocessor:
    def __init__(self, config:PreProcessorConfig):
        self.config=config
        self.test_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=config.IMAGE_SIZE),
                A.PadIfNeeded(min_height=config.IMAGE_SIZE,min_width=config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=config.MEAN, std=config.STD, max_pixel_value=255),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format='coco',
                min_visibility=0.4,
                label_fields=['labels']
            )
        )
        self.train_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=config.IMAGE_SIZE),
                A.PadIfNeeded(min_height=config.IMAGE_SIZE,min_width=config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ColorJitter(brightness=config.BRIGHTNESS, contrast=config.CONTRAST,
                              saturation=config.SATURATION, hue=config.HUE, p=config.PROB),
                A.HorizontalFlip(p=config.PROB),
                A.Normalize(mean=config.MEAN, std=config.STD, max_pixel_value=255),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format='coco',
                min_visibility=0.4,
                label_fields=['labels']
            )
        )
    
    def save(self):
        joblib.dump(self.test_transform, self.config.ARTIFACT_DIR + '/train_transform.pkl')
        joblib.dump(self.test_transform, self.config.ARTIFACT_DIR + '/test_transform.pkl')

        logger.info(f"Preprocessor saved at {self.config.ARTIFACT_DIR}")