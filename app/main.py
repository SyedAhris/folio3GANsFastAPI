from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware


import model as model
import firebase as fb
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://urdugan-fd007.web.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#initializing the model
generator = model.get_model()
#initialize pyrebase
firebase = fb.initFirebase()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/categories/")
async def read_item():
    cats = model.getCategories()
    return {'categories':cats}

@app.get("/predict/")
async def read_item(num_of_examples: str, label: str, ip: str):
    #model.get_images(num_of_examples=num_of_examples, label=label, gen=generator)
    directory = fb.uploadImages(firebase, ip)
    return  {'msg':'success, files uploaded to bucket','folderName':directory}
