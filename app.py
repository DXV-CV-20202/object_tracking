# import the necessary packages
from tracker import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2


original_frame = None
keyed_frame = None
tracked_frame = None

lock = threading.Lock()
app = Flask(__name__)

vs = VideoStream().start()
time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")


def track_object(frame_count):
    global vs, original_frame, keyed_frame, tracked_frame

    # Basically, you just need to assign the processed value to the variables
    # original_frame, keyed_frame and tracked_frame for this to work

    md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    while True:
        frame = vs.read()
        original_frame = frame.copy()
        keyed_frame = frame.copy()

        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frame_count:
            # detect motion in the image
            motion = md.detect(gray)
            # check to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                    (0, 0, 255), 2)
        
        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

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
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=track_object, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()