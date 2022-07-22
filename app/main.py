from fastapi import FastAPI, Response
import model as model
app = FastAPI()

#initializing the model
generator = model.get_model()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/predict/")
async def read_item(num_of_examples: int, label: int):
    img  = model.get_images(num_of_examples=num_of_examples, label=label, gen=generator)
    return Response(content=img, media_type="image/png")