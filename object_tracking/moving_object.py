from collections import namedtuple
from .filters import KalmanTracking

class MovingObject:
    def __init__(self, _id, bbox, keypoints, descriptions, color):
        self._id = _id
        self.bbox = bbox
        self.keypoints = keypoints
        self.descriptions = descriptions
        self.tracking = [bbox]
        self.unseen_time = 0
        self.kalman_tracking = KalmanTracking(bbox)
        self.color = color

    def __str__(self):
        return str(namedtuple('MovingObject', ['id', 'unseen_time', 'bbox'])(self._id, self.unseen_time, self.bbox))

    def __repr__(self):
        return self.__str__()