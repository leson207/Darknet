from src.entity.configuration import ConfigManager
from src.component.DataIngestor import DataIngestor
from src.component.Preprocessor import Preprocessor
from src.component.DataModule import DataModule
from src.component.Trainer import Trainer
from src.component.Evaluator import Evaluator
from src.component.Predictor import Predictor

from PIL import Image

if __name__=="__main__":
    config_manager = ConfigManager()

    data_ingestor_config=config_manager.get_data_ingestor_config()
    data_ingestor=DataIngestor(data_ingestor_config)
    data_ingestor.create_dataset()
    data_ingestor.create_label_mapping()
    # data_ingestor.sync_to_s3()
    # data_ingestor.sync_from_s3()

    preprocessor_config=config_manager.get_preprocessor_config()
    preprocessor=Preprocessor(preprocessor_config)
    preprocessor.save()

    data_module_config=config_manager.get_data_module_config()
    data_module = DataModule(data_module_config)
    loader=data_module.get_data_loader('test',4)
    test_batch=next(iter(loader))
    print(test_batch[0].shape)
    for x in test_batch[1]:
        print(x.shape)

    trainer_config = config_manager.get_trainer_config()
    trainer = Trainer(trainer_config)
    trainer.train()
    trainer.evaluate(use_map=False)

    evaluator_config=config_manager.get_evaluator_config()
    evaluator=Evaluator(evaluator_config)
    evaluator.eval(use_map=False)

    predictor_config=config_manager.get_predictor_config()
    predictor=Predictor(predictor_config)
    img=Image.open('input.png')
    predictor.inference(img).show()