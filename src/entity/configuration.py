from src.entity.entity import *
from src.utils.common import *

class ConfigManager:
    def __init__(self, config_file_path='params.yaml'):
        self.config= read_yaml(config_file_path)
    
    def get_data_ingestor_config(self):
        config=self.config.data_ingestor_config
        create_directories([config.ARTIFACT_DIR])
        data_ingestor_config = DataIngestorConfig(**config)
        return data_ingestor_config

    def get_preprocessor_config(self):
        config=self.config.preprocessor_config
        config.IMAGE_SIZE=self.config.IMAGE_SIZE
        create_directories([config.ARTIFACT_DIR])
        preprocessor_config = PreProcessorConfig(**config)
        return preprocessor_config

    def get_data_module_config(self):
        config=self.config.data_module_config
        config.IMAGE_SIZE=self.config.IMAGE_SIZE
        data_module_config = DataModuleConfig(**config)
        data_module_config.ANCHORS=[[tuple(box) for box in scale] for scale in self.config.ANCHORS]
        return data_module_config

    def get_loss_config(self):
        config=self.config.loss_config
        config.ANCHORS=[[tuple(box) for box in scale] for scale in self.config.ANCHORS]
        loss_config=LossConfig(**config)
        return loss_config

    def get_trainer_config(self):
        config=self.config.trainer_config

        config.DATA_MODULE_CONFIG=self.get_data_module_config()
        config.LOSS_CONFIG=self.get_loss_config()

        config.MODEL_CONFIG=ModelConfig(**self.config.model_config)
        config.MODEL_CONFIG.NUM_CLASSES=self.config.NUM_CLASSES

        config.TRAINING_CONFIG=TrainingConfig(**self.config.training_config)
        config.MAP_CONFIG=mAPConfig(**self.config.map_config)
        config.MAP_CONFIG.NUM_CLASSES=self.config.NUM_CLASSES
        config.MAP_CONFIG.IMAGE_SIZE=self.config.IMAGE_SIZE
        config.MAP_CONFIG.ANCHORS=[[tuple(box) for box in scale] for scale in self.config.ANCHORS]
        
        create_directories([config.ARTIFACT_DIR])
        trainer_config=TrainerConfig(**config)
        return trainer_config

    def get_evaluator_config(self):
        config=self.config.evaluator_config
        config.DATA_MODULE_CONFIG=self.get_data_module_config()
        config.LOSS_CONFIG=self.get_loss_config()   
        config.MAP_CONFIG=mAPConfig(**self.config.map_config)
        config.MAP_CONFIG.NUM_CLASSES=self.config.NUM_CLASSES
        config.MAP_CONFIG.IMAGE_SIZE=self.config.IMAGE_SIZE
        config.MAP_CONFIG.ANCHORS=[[tuple(box) for box in scale] for scale in self.config.ANCHORS]
        evaluator_config=EvaluatorConfig(**config)
        return evaluator_config

    def get_predictor_config(self):
        config=self.config.predictor_config
        config.ANCHORS=[[tuple(box) for box in scale] for scale in self.config.ANCHORS]
        config.GRID_SIZES=[self.config.IMAGE_SIZE//32, self.config.IMAGE_SIZE//16, self.config.IMAGE_SIZE//8]
        return PredictorConfig(**config)

    def get_inferencer_config(self):
        config=self.config.inferencer_config
        config.ANCHORS=[[tuple(box) for box in scale] for scale in self.config.ANCHORS]
        config.GRID_SIZES=[self.config.IMAGE_SIZE//32, self.config.IMAGE_SIZE//16, self.config.IMAGE_SIZE//8]
        config.PREPROCESSOR_CONFIG=self.get_preprocessor_config()
        return InferencerConfig(**config)