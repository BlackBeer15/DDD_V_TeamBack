from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import imutils
import cv2
from pathlib import Path
import os
import imutils
from PIL import Image
import requests
import json

url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
print()
def sendQuestion(quest):
    payload = json.dumps({
    "model": "GigaChat",
    "messages": [
        {
        "role": "user",
        "content": question
        }
    ],
    "stream": False,
    "repetition_penalty": 1
    })
    
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': 'Bearer MjcwM2NlNGYtNmI2Mi00OThjLWExNjctZmJlYzE5NTBlMmEyOjdmZTJkMjQyLWEzOGMtNGY4OS1hYjk4LThhZGY5NDJiZDVjMA=='
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response

CLASSES = ['Дом Губкиных']
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
dir = Path('.')
path=os.path.join(os.path.abspath(os.curdir) , 'model.onnx')
PATH = os.path.join(dir, "model.pth")
net.load_state_dict(torch.load(PATH))

net = cv2.dnn.readNetFromONNX (path)

def getDescByImage(path):
    place = {}
    frame = np.array(Image.open(path))
    frame = imutils.resize(frame, width=400)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (32, 32)),scalefactor=1.0/32
                              , size=(32, 32), mean= (128,128,128), swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    class_mark=np.argmax(detections)

    place['place'] = CLASSES[class_mark-1]
    place['architecture'] = sendQuestion(f"Кто архитектор {CLASSES[class_mark-1]}")
    place['year'] = sendQuestion(f"Какой год постройки у {CLASSES[class_mark-1]}")
    place['history'] = sendQuestion(f"Сделай историческую справку про {CLASSES[class_mark-1]}")
    return place
app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path=os.path.join(os.path.abspath(os.curdir) , 'my_model.onnx')
@app.post("/getByImage")
async def upload_file(photoTower: UploadFile):
    try:
        contents = photoTower.file.read()
        with open(photoTower.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        photoTower.file.close()
    place = getPlaceByImage(photoTower.filename)
    os.remove(photoTower.filename)
    return place

@app.post("/getByAddress")
def getByAddress(data  = Body()):
    return {"place" : "Дом Губкиных", "architecture" : "П.А. Парамонова", "year" : "1918", "address": "Российская Федерация, город Кемерово, улица Шестакова, дом 60", "history" : "Дом семьи Губкиных в город Шегловск (с 1932 г. – Кемерово) был перевезен в 1926 году из села Салтымаково, остов был сплавлен по реке Томь. Род Губкиных занимался сплавом леса по реке Томи, в годы гражданской войны прославились партизанской деятельностью."}

@app.get("/info")
def upload_file():
  return "{ year : "",  fio : "",  history : "" }"

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)