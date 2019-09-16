import cv2

ACTION_MAPPING = {0: 'Abnormal', 1: 'Normal'}

class VideoAnalyzer:

    def __init__(self, src="video.mp4"):
        cap = cv2.VideoCapture(src)
        _, frame = cap.read()

    def add_predict(self, frame, predict):
        color = (0, 255, 0)
        if predict == 0:
            color = (0, 0, 255)
        height, width, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (width, height), color, 2)
        cv2.putText(frame, ACTION_MAPPING[predict], (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2, lineType=2)

