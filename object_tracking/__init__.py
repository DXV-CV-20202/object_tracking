from __future__ import print_function

import cv2 as cv
import imutils
import numpy as np

from .feature_extractor import SIFT
from .match_object import match_object
from .moving_object import MovingObject

class MotionDetector:
    def __init__(self, algorithm='MOG2'):
        self.algorithm = algorithm
        if self.algorithm == 'MOG2':
            self.backSub = cv.createBackgroundSubtractorMOG2(varThreshold=24)
        else:
            self.backSub = cv.createBackgroundSubtractorKNN()

        self.disappear_threshold = 10
        self.num_keypoints = 16

        self.extractor = SIFT()
        self.moving_objects = []
        self.all_objects = set()
        self._id = 0


    def analyse(self, frame):
        fgMask = self.backSub.apply(frame)

        thresh = cv.dilate(fgMask, None, iterations=0)
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        detected_objects = []

        for c in cnts:
            if cv.contourArea(c) < 400:
                continue
            (x, y, w, h) = cv.boundingRect(c)
            patch = frame[y:y+h, x:x+w]
            if np.prod(patch.shape) <= 0:
                continue
            keypoints, descriptions = self.extractor.extract_full(patch)
            if type(descriptions) == type(None):
                continue
            keypoint_description = list(zip(keypoints, descriptions))
            keypoint_description.sort(key=lambda x:x[0].response, reverse=True)
            keypoints = [kd[0] for kd in keypoint_description[:self.num_keypoints]]
            descriptions = np.array([kd[1] for kd in keypoint_description[:self.num_keypoints]])
            color1 = (list(np.random.choice(range(256), size=3)))
            color =[int(color1[0]), int(color1[1]), int(color1[2])]
            detected_objects.append(MovingObject(self._id, (x, y, x + w, y + h), keypoints, descriptions, color))
            self._id += 1
            self.all_objects.add(detected_objects[-1])

        matching = []
        if len(self.moving_objects) > 0:
            matching = match_object(self.moving_objects, detected_objects)

        unseen_set = set(range(len(self.moving_objects)))
        firstseen_set = set(range(len(detected_objects)))

        for m in matching:
            mo = self.moving_objects[m[0]]
            do = detected_objects[m[1]]
            mo.kalman_tracking.predict()
            bbox = mo.kalman_tracking.correct(bbox=do.bbox, flag=True)[0]
            mo.bbox = bbox
            mo.keypoints = do.keypoints
            mo.descriptions = do.descriptions
            mo.tracking.append(bbox)
            mo.unseen_time = 0
            unseen_set.remove(m[0])
            firstseen_set.remove(m[1])

        for unseen in unseen_set:
            mo = self.moving_objects[unseen]
            mo.kalman_tracking.predict()
            bbox = mo.kalman_tracking.correct(bbox=None, flag=False)[0]
            mo.bbox = bbox
            mo.tracking.append(bbox)
            mo.unseen_time += 1

        moving_objects = list(filter(lambda x : x.unseen_time <= self.disappear_threshold, self.moving_objects))
        
        for firstseen in firstseen_set:
            moving_objects.append(detected_objects[firstseen])

        return thresh, moving_objects