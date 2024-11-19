import torch
from tqdm.auto import tqdm

from src.logger import logger
from src.entity.entity import EvaluatorConfig
from src.component.DataModule import DataModule
from src.component.Criterion import YOLOLoss
from src.component.mAP import MAP

class Evaluator:
    def __init__(self, config:EvaluatorConfig):
        self.config=config
        dataset=DataModule(config.DATA_MODULE_CONFIG)
        self.test_loader=dataset.get_data_loader('test', batch_size=config.BATCH_SIZE)
        self.criterion=YOLOLoss(config.LOSS_CONFIG)

        self.models={}
        model=torch.load(config.MODEL_PATH,weights_only=False)
        self.models['model']=model

        model=torch.jit.load(config.MODEL_SCRIPTED_PATH)
        self.models['scripted_model']=model
    
    def eval(self, use_map=True):
        for name, model in self.models.items():
            self.single_eval(model, use_map)
            logger.info(f'Finnish Evaluation for {name}')

    def single_eval(self, model, use_map=True):        
        model.eval()
        if use_map:
            mAP=MAP(self.config.MAP_CONFIG)
        p_bar=tqdm(self.test_loader, desc='Evaluation: ')
        for imgs, targets in p_bar:
            with torch.no_grad():
                pred=model(imgs)
                loss=self.criterion(pred,targets).item()
            
            if use_map:
                mAP.update(pred, targets)
            p_bar.set_postfix(loss=loss)

        if use_map:
            result=mAP.compute()
            print(result)
    