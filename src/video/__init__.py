import cv2
import os

ACTION_MAPPING = {0: 'Abnormal', 1: 'Normal'}
FRAME_INTERVAL = 25

class VideoAnalyzer:

    def __init__(self, model, src=None):
        if src is None:
            src = "{}/video/video.mp4".format(os.getcwd());
        self.cap = cv2.VideoCapture(src)
        self.model = model

    def add_predict(self, frame, predict):
        color = (0, 255, 0)
        if predict == 0:
            color = (0, 0, 255)
        height, width, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (width, height), color, 2)
        cv2.putText(frame, ACTION_MAPPING[predict], (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2, lineType=2)

    def stream_analysis(self):
        _, frame = self.cap.read()
        yield frame
        segment = []
        predict = None
        ret = True
        frame_count = 0
        while ret:
            print(frame_count)
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                image = cv2.resize(frame, (331, 331), interpolation=cv2.INTER_AREA)
                segment.append(image)
                frame_count += 1
            except Exception as e:
                continue
            if len(segment) == FRAME_INTERVAL:
                predict = self.model.predict(segment)
                print(predict)
                del segment[:]
            if predict is not None and frame_count % FRAME_INTERVAL < 6:
                self.add_predict(frame, predict)
            yield frame
        # When everything is done, release the capture
        self.cap.release()

    def get_byte_stream(self):
        for frame in self.stream_analysis():
            yield bytearray(cv2.imencode(".jpg", frame)[1].tostring())

    def write_to_video(self):
        st = self.stream_analysis()
        frame = next(st)
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        output_video = cv2.VideoWriter('output.avi', fourcc, 20.0,
                                    (width, height))
        for frame in st:
            output_video.write(frame)
        output_video.release()
        cv2.destroyAllWindows()
