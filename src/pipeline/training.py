from src.entity.configuration import ConfigManager
from src.component.DataModule import DataModule
from src.component.Trainer import Trainer

if __name__=="__main__":
    config_manager = ConfigManager()

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
    # trainer.evaluate()