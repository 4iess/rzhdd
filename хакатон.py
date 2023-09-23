from imageai.Detection import ObjectDetection
import cv2
import os
import numpy
import time
from voice import object_voice, image_counter
import asyncio
from threading import Thread


exe_path = os.getcwd()

camera = cv2.VideoCapture(0)

check_array = [
    'traffic light',
    'person',
    'car'
]

# def ident_person(array):
#     for obj in array:
#         if obj['name'] in check_array and array is not None:
#             return f"{obj['name']}"


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(exe_path, 'yolov3.pt'))
detector.loadModel()
finish = 0


while camera.isOpened():
    ret, frame = camera.read()

    start = time.time()
    if start - finish > 1:
        _, array_detection = detector.detectObjectsFromImage(
            input_image=frame,
            output_type='array',
            minimum_percentage_probability=30
        )

        for obj in array_detection:
            if obj['name'] in check_array:
                image_counter += 1
                cv2.imwrite(f'output_image_{image_counter}.png', frame)
                object_voice(object_name=obj['name'])
                cv2.rectangle(frame, (obj['box_points'][0], obj['box_points'][1]), (obj['box_points'][2], obj['box_points'][3]), color=(0, 0, 255), thickness=2)
                cv2.putText(frame, f"{obj['box_points']}", (obj['box_points'][0], obj['box_points'][1]-10), cv2.FONT_ITALIC, 0.5, obj['box_points'], 2)
                finish = time.time()

    cv2.imshow('hakaton', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()