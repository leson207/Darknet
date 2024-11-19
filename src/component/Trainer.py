import torch
import mlflow
import bentoml
from tqdm.auto import tqdm
from torchinfo import summary


from src.component.DataModule import DataModule
from src.component.Model import YoloMimic
from src.component.Criterion import YOLOLoss
from src.component.Metric import Metric
from src.component.mAP import MAP

from src.logger import logger
from src.entity.entity import TrainerConfig


device='cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    def __init__(self, config:TrainerConfig):
        self.config=config
        self.data_module = DataModule(config.DATA_MODULE_CONFIG)
        self.criterion=YOLOLoss(config.LOSS_CONFIG)
        self.model=YoloMimic(3,208).to(device)
        self.opt=torch.optim.Adam(params = self.model.parameters(), lr=1e-3)
        self.metric=Metric()

    def train_step(self, inputs, targets):
        inputs = inputs.to(device)
        targets = [target.to(device) for target in targets]
        preds = self.model(inputs)
        loss= self.criterion(preds, targets)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def val_step(self, inputs, targets):
        inputs = inputs.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            preds = self.model(inputs)
            loss= self.criterion(preds, targets)

        return loss.item()

    def evaluate(self, use_map=True):
        if use_map:
            mAP=MAP(self.config.MAP_CONFIG)
        test_loader = self.data_module.get_data_loader('test', batch_size=self.config.TRAINING_CONFIG.BATCH_SIZE, shuffle=False)
        p_bar = tqdm(test_loader, desc='Evaluation: ')
        self.model.eval()
        for inputs, targets in p_bar:
            inputs = inputs.to(device)
            targets = [target.to(device) for target in targets]
            with torch.no_grad():
                preds = self.model(inputs)
                loss= self.criterion(preds, targets).item()
            
            if use_map:
                mAP.update(preds, targets)
            self.metric.update({'train_loss': loss})
            p_bar.set_postfix(train_loss=loss)
        
        if use_map:
            result=mAP.compute()
            print(result)
            return result

    def train(self):
        num_epochs=self.config.TRAINING_CONFIG.NUM_EPOCHS
        train_loader = self.data_module.get_data_loader('train', batch_size=self.config.TRAINING_CONFIG.BATCH_SIZE, shuffle=True)
        val_loader = self.data_module.get_data_loader('validation', batch_size=self.config.TRAINING_CONFIG.BATCH_SIZE, shuffle=False)
        logger.info('Start model trainning')

        # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("Darknet")
        with mlflow.start_run():
            params = {
                "epochs": num_epochs,
                "learning_rate": self.config.TRAINING_CONFIG.LEARNING_RATE,
                "batch_size": self.config.TRAINING_CONFIG.BATCH_SIZE,
                "optimizer": "Adam",
            }
            mlflow.log_params(params)

            for epoch in tqdm(range(num_epochs)):
                self.model.train()
                p_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}: ')
                for inputs, targets in p_bar:
                    loss=self.train_step(inputs, targets)

                    self.metric.update({'train_loss': loss})
                    mlflow.log_metric('train_loss', loss, step=len(train_loader)*epoch+p_bar.n)
                    p_bar.set_postfix(train_loss=loss)

                    if p_bar.n>=10:
                        break

                self.model.eval()
                p_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}: ')
                for inputs, targets in p_bar:
                    loss=self.val_step(inputs, targets)

                    self.metric.update({'val_loss': loss})
                    mlflow.log_metric('val_loss', loss, step=len(val_loader)*epoch+p_bar.n)
                    p_bar.set_postfix(val_loss=loss)

                self.metric.finalize()
            
            with open(self.config.ARTIFACT_DIR+"/model_summary.txt", "w") as f:
                f.write(str(summary(self.model)))
            mlflow.log_artifact(self.config.ARTIFACT_DIR+"/model_summary.txt")

            mlflow.pytorch.log_model(self.model, "model")

        logger.info('Finished model trainning')
        self.save(self.data_module.collator.test_transform)

    def save(self, transform=None):
        self.model.eval()
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(self.config.ARTIFACT_DIR+'/model_scripted.pt')
        torch.save(self.model, self.config.ARTIFACT_DIR+'/model.pt')
        bentoml.pytorch.save_model(
            name='model',
            model=self.model,
            custom_objects={
                'test_transformation': transform
            }
        )
        logger.info('Model saved')