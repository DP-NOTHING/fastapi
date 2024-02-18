from fastapi import FastAPI, File, UploadFile
from typing import Union
from pydantic import BaseModel
from yolo5face.get_model import get_model
from deepface import DeepFace
import cv2
import numpy as np

model = get_model("yolov5n", device=0, min_face=24)
app = FastAPI()
    
@app.post("/check")
def check(file: UploadFile = File(...), file2: UploadFile = File(...)):
    contents1 = file.file.read()
    contents = cv2.imdecode(np.fromstring(contents1, np.uint8), cv2.IMREAD_COLOR)
    contents = cv2.cvtColor(contents, cv2.COLOR_BGR2RGB)
    enhanced_boxes, enhanced_key_points, enhanced_scores = model(contents, target_size=[320, 640, 1280])
    nop = len(enhanced_boxes)
    
    pass1 = file2.file.read()
    pass2 = cv2.imdecode(np.fromstring(pass1, np.uint8), cv2.IMREAD_COLOR)
    veri=DeepFace.verify(contents, pass2,detector_backend="opencv",model_name="Facenet512",enforce_detection=False,align=True,distance_metric="euclidean_l2")
    print(veri)
    return {"no_of_person": nop, "verified": veri['verified']}



