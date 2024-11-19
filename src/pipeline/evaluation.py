from src.entity.configuration import ConfigManager
from src.component.Evaluator import Evaluator

if __name__=="__main__":
    config_manager = ConfigManager()

    evaluator_config=config_manager.get_evaluator_config()
    evaluator=Evaluator(evaluator_config)
    evaluator.eval(use_map=False)