import os
import cv2
import time

from model import Model, FRAME_INTERVAL

md = Model()


def add_predict(frame, predict):
    action_mapping = {0: 'Abnormal', 1: 'Normal'}
    color = (0, 255, 0)
    if predict == 0:
        color = (0, 0, 255)
    height, width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (width, height), color, 2)
    cv2.putText(frame, action_mapping[predict], (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2, lineType=2)


fps_display_interval = 5  # frames
frame_count = 0

cap = cv2.VideoCapture('video.mp4')
_, frame = cap.read()

segment = []

height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
output_video = cv2.VideoWriter('output.avi', fourcc, 20.0,
                               (width, height))
predict = None
ret = True
frame_count = 0
while ret:
    ret, frame = cap.read()
    if not ret:
        continue
    try:
        image = cv2.resize(frame, (331, 331), interpolation=cv2.INTER_AREA)
        segment.append(image)
        frame_count += 1
    except Exception as e:
        continue

    if len(segment) == FRAME_INTERVAL:
        predict = md.predict(segment)
    if predict is not None and frame_count % FRAME_INTERVAL < 6:
        add_predict(frame, predict)
    output_video.write(frame)

# When everything is done, release the capture
cap.release()
output_video.release()
cv2.destroyAllWindows()
