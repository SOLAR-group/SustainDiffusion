import os
import torch
import numpy as np
from ultralytics import YOLO
import cv2

if "yolov8n_train.pt" not in os.listdir():
    print("Train YOLO")
    model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n detection model
    model.to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # use cuda if you have a good GPU, otherwise CPU
    model.train(data="coco128.yaml", epochs=3)  # train the model
    model.save("yolov8n_train.pt")  # save the model
else:
    print("Load YOLO")
    model = YOLO("yolov8n_train.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype="uint8")
thickness = 2
fontScale = 0.5


# Functions to compute image quality with YOLO
def read_box(box):
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = model.names[box.cls[0].item()]
    conf = round(box.conf[0].item(), 2)
    return [class_id, cords, conf]


def addBoxesImage(currentImage, boxesInfo):
    image = cv2.imread(currentImage)
    for box in boxesInfo:
        class_id = box[0]
        confidence = box[2]
        color = [int(c) for c in colors[list(model.names.values()).index(class_id)]]
        #        color = colors[list(model.names.values()).index(class_id)]
        cv2.rectangle(
            image,
            (box[1][0], box[1][1]),
            (box[1][2], box[1][3]),
            color=color,
            thickness=thickness,
        )
        text = f"{class_id}: {confidence:.2f}"
        (text_width, text_height) = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, thickness=thickness
        )[0]
        text_offset_x = box[1][0]
        text_offset_y = box[1][1] - 5
        box_coords = (
            (text_offset_x, text_offset_y),
            (text_offset_x + text_width + 2, text_offset_y - text_height),
        )
        overlay = image.copy()
        cv2.rectangle(
            overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED
        )
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        cv2.putText(
            image,
            text,
            (box[1][0], box[1][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(0, 0, 0),
            thickness=thickness,
        )
    cv2.imwrite(currentImage + "_yolo8.png", image)


def img2text(image_path):
    result = model(image_path)  # predict on an image
    boxesInfo = []
    counting = {}
    for box in result[0].boxes:
        currentBox = read_box(box)
        boxesInfo.append(currentBox)
        if currentBox[0] in counting.keys():
            counting[currentBox[0]] += 1
        else:
            counting[currentBox[0]] = 1
    return counting, boxesInfo
