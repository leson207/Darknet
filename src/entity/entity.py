from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DataIngestorConfig:
    ARTIFACT_DIR: str
    DATASET_NAME: str
    BUCKET_NAME: str
    CLOUD_DIR: str

@dataclass
class PreProcessorConfig:
    ARTIFACT_DIR: str
    IMAGE_SIZE: int
    MEAN: List[float]
    STD: List[float]
    BRIGHTNESS: float
    CONTRAST: float
    SATURATION: float
    HUE: float
    PROB: float

@dataclass
class DataModuleConfig:
    DATA_DIR: str
    PREPROCESSOR_DIR: str
    TRAIN_SIZE: float
    IMAGE_SIZE: int
    ANCHORS: List[List[Tuple[float, float]]]

@dataclass
class LossConfig:
    ANCHORS: List[List[Tuple[float, float]]]
    LAMBDA_OBJ: float
    LAMBDA_NOOBJ: float
    LAMBDA_COORD: float
    LAMBDA_CLASS: float

@dataclass
class ModelConfig:
    IN_CHANNELS: int
    NUM_CLASSES: int

@dataclass
class TrainingConfig:
    NUM_EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float

@dataclass
class mAPConfig:
    EPS: float
    ANCHORS: List
    NUM_CLASSES: int
    IMAGE_SIZE: int
    IOU_THRESHOLD: List

@dataclass
class TrainerConfig:
    ARTIFACT_DIR: str
    DATA_MODULE_CONFIG: DataModuleConfig
    LOSS_CONFIG: LossConfig
    MODEL_CONFIG: ModelConfig
    TRAINING_CONFIG: TrainingConfig
    MAP_CONFIG :mAPConfig

@dataclass
class EvaluatorConfig:
    DATA_MODULE_CONFIG: DataModuleConfig
    BATCH_SIZE: int
    LOSS_CONFIG: LossConfig
    MAP_CONFIG: mAPConfig
    MODEL_PATH: str
    MODEL_SCRIPTED_PATH: str

@dataclass
class PredictorConfig:
    DATA_DIR: str
    MODEL_PATH: str
    TRANSFORM_PATH: str
    ANCHORS: List
    GRID_SIZES: List

@dataclass
class InferencerConfig:
    DATA_DIR: str
    ANCHORS: List
    GRID_SIZES: List
    PREPROCESSOR_CONFIG: PreProcessorConfig