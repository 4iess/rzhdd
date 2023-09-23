import logging
import csv
import numpy
import os
import cv2
import time
import asyncio
from imageai.Detection import ObjectDetection
from voice import object_voice
import datetime
data = [
    ['filename', 'event', 'timestamps']
]
mas_str = []
count = 0
print("С какой камеры вы будете использовать видео?(1.Боковая, 2.Курсовая")
mesto = int(input())
if mesto == 2:
    x_1, y_1, x_2, y_2 = 400, 150, 1200, 1500
elif mesto == 1:
    x_1, y_1, x_2, y_2 = 0, 0, 500, 1500
some_bytes = b'\x01\x02'
logger = logging.getLogger()
exe_path = os.getcwd()
video_file_path = "02_35_34.mp4"  # Укажите путь к вашему видчеофайлуй
check_array = ['traffic light', 'person', 'car']


x1, y1, x2, y2 = 0, 0, 0, 0

async def process_frame(detector, frame, timestamp): #
    _, array_detection = await asyncio.to_thread(
        detector.detectObjectsFromImage,
        input_image=frame,
        output_type='array',
        minimum_percentage_probability=30
    )

    for obj in array_detection:
        if obj['name'] in check_array:
            global x1, y1, x2, y2, count, current_time, sek, mas_str
            x1, y1, x2, y2 = obj['box_points']
            filename = datetime.datetime.now().strftime("%d.%m.%Y %H-%M-%S") + ".png"
            cv2.imwrite(f'output_image_{filename}', frame)
            await object_voice(obj['name'])
            count += 1
            mas_str.append(int(timestamp))


async def main():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(exe_path, 'yolov3.pt'))
    detector.loadModel()
    finish = 0


    # Открываем видеофайл для чтения
    video_capture = cv2.VideoCapture(video_file_path)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        area = frame[y_1:y_2, x_1:x_2]
        current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not ret:
            break

        start = time.time()
        if start - finish > 1.5:
            asyncio.create_task(process_frame(detector, area, timestamp=current_time))

            finish = time.time()

        cv2.rectangle(area, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('hakaton', area)
        await asyncio.sleep(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    global data, mas_str
    mas_str = [':'.join(str(datetime.timedelta(seconds=x)).split(':')[1:]) for x in mas_str]
    data = numpy.append(data, [[video_file_path,count,str(mas_str)]], axis=0)
    with open('sw_data_new.csv', 'a', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data)
if __name__ == "__main__":
    asyncio.run(main())
