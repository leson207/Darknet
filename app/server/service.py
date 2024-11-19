import numpy as np
from PIL import Image as PILImage

import bentoml
from bentoml.io import Multipart, JSON, Image

from app.server.Inferencer import Inferencer
from src.entity.configuration import ConfigManager


config_manager = ConfigManager()
config=config_manager.get_inferencer_config()
inferencer=Inferencer(config)

model=bentoml.pytorch.get("model:latest")
model_runner = model.to_runner()
# transform = model.custom_objects['test_transformation'] # can't use in dockerize

svc = bentoml.Service("object_detection_service", runners=[model_runner])
@svc.api(input=Multipart(image=Image()), output=JSON())
async def predict(image: PILImage.Image):
    input = inferencer.transform(image=np.array(image), labels=[])['image'][None]

    outputs = await svc.runners[0].async_run(input)
    
    print('*'*50)
    print(type(outputs))
    print(type(outputs[0]))
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    print('*'*50)
    print('A'*50)
    
    return np.array(inferencer.post_process(image, input, outputs))
