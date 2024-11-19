from PIL import Image
import bentoml
import io

image = Image.open('input.png')

image_bytes = io.BytesIO()
image.save(image_bytes, format='PNG')
image_bytes.seek(0)

with bentoml.client.SyncHTTPClient.from_url("http://localhost:3000") as client:
    response = client.call(
        "predict",
        image=image
    )

print(response)