import io
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from bentoml.client import AsyncHTTPClient

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import prometheus_client

app = FastAPI()
app.mount("/metrics", prometheus_client.make_asgi_app())
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount("/static", StaticFiles(directory="app/client/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=open("app/client/template/index.html").read())

@app.post("/predict")
async def predict_route(request: dict=Body(...)):
    b64_image=request['image']
    image = base64.b64decode(b64_image)
    image = io.BytesIO(image)
    image = Image.open(image)

    # client = await AsyncHTTPClient.from_url("http://localhost:3000")
    client = await AsyncHTTPClient.from_url("http://bentoml_service:3000") # use for docker
    # client = await AsyncHTTPClient.from_url("http://host.docker.internal:3000") # use for docker

    response = await client.call("predict", image=image)

    image = Image.fromarray(np.array(response, dtype=np.uint8))    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return [img_base64]