import cv2
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status,Depends,Request
from fastapi.responses import JSONResponse, Response
from functools import cache
from starlette.middleware.cors import CORSMiddleware
from src.llm_service import TemplateLLM, get_llm_service
from src.prompts import ProjectParams
from src.parsers import ProjectIdeas
from src.config import get_settings
from src.detector import ObjectDetector, Detection

_SETTINGS = get_settings()

app = FastAPI(
    title=_SETTINGS.service_name,
    version=_SETTINGS.k_revision
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@cache
def get_object_detector():
    print("creating model...")
    return ObjectDetector()

def predict_uploadfile(predictor, file, threshold):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Not an image"
        )
    # convertir a una imagen de Pillow
    img_obj = Image.open(img_stream)
    # crear array de numpy
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array, threshold), img_array

def annotate_image(image_array, prediction: Detection):
    ann_color = (255, 255, 0)
    annotated_img = image_array.copy()
    for box, label, conf in zip(prediction.boxes, prediction.labels, prediction.confidences):
        cv2.rectangle(annotated_img, (box[0], box[1]), (box[2], box[3]), ann_color, 3)
        cv2.putText(
            annotated_img, 
            label, 
            (box[0], box[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
        cv2.putText(
            annotated_img, 
            f"{conf:.1f}", 
            (box[0], box[3] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    ## annotation
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)

    return image_stream

@app.post("/generate")
def generate_project(params: ProjectParams, service: TemplateLLM = Depends(get_llm_service)) -> ProjectIdeas:
    return service.generate(params)


@app.get("/")
def root():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app")
# crear un endpoint que reciva una imagen y devuelva la cantidad de personas que se detecta en la imagen
