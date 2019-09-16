import os
import cv2
import time
from flask import Flask, Response, request

from model import Model
from video import VideoAnalyzer

md = Model()

vd = VideoAnalyzer(model=md)

vd.write_to_video()
# app = Flask(__name__)

# def gen(delay=0.02):
#     for frame in vd.get_byte_stream():
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         time.sleep(delay)

# @app.route("/stream")
# def stream():
#     dl = 0
#     if request.args.get("delay") == "1":
#         dl = 0.02
#     return Response(gen(dl), mimetype='multipart/x-mixed-replace; boundary=frame')

# app.run(port=3000)
