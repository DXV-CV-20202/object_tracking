# import the necessary packages
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import time
import cv2
from object_tracking import MotionDetector
import datetime
import imutils

original_frame = None
keyed_frame = None
tracked_frame = None

lock = threading.Lock()
app = Flask(__name__)

time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")


def track_object():
    global vs, original_frame, keyed_frame, tracked_frame

    # Basically, you just need to assign the processed value to the variables
    # original_frame, keyed_frame and tracked_frame for this to work

    md = MotionDetector()
    while True:
        ret, frame = vs.read()
        original_frame = frame.copy()

        thresh, moving_objects = md.analyse(frame)
        keyed_frame = thresh
        for obj in moving_objects:
            if obj.unseen_time > 0:
                continue
            xLeft, yTop, xRight, yBottom = [int(c) for c in obj.bbox]
            cv2.rectangle(frame, (xLeft, yTop), (xRight, yBottom), obj.color, 2)
            cv2.rectangle(frame, (xLeft, yTop), (xRight, yTop + 20), obj.color, -1)
            cv2.putText(frame, str(obj._id), (xLeft, yTop + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (0,0,0), thickness=2)
        
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(vs.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        tracked_frame = frame.copy()



def generate_frame(frame_type):
    global original_frame, keyed_frame, tracked_frame, lock
    output_frame = None
    while True:
        if frame_type == 'original':
            output_frame = original_frame
        elif frame_type == 'keyed':
            output_frame = keyed_frame
        elif frame_type == 'tracked':
            output_frame = tracked_frame
        else:
            raise Exception('Unknown frame type')

        if output_frame is None:
            continue

        (flag, encoded_img) = cv2.imencode(".jpg", output_frame)
        if not flag:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_img) + b'\r\n')


@app.route("/video_feed/<frame_type>")
def video_feed(frame_type):
    return Response(generate_frame(frame_type),
            mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default='localhost',
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=5000,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='videos/bolt-multi-size-detection.mp4')
    ap.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = vars(ap.parse_args())

    vs = cv2.VideoCapture(cv2.samples.findFileOrKeep(args["input"]))
    t = threading.Thread(target=track_object)
    t.daemon = True
    t.start()
    if not vs.isOpened():
        print('Unable to open: ' + args["input"])
        exit(0)

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
