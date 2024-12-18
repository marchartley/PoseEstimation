import numpy as np
import cv2
from numpy.core.defchararray import lower


def parseYoloResults(frame, out, threshold = 0.1):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    objects = []
    for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > threshold:
            center_x = int(detection[0] * frameWidth)
            center_y = int(detection[1] * frameHeight)
            width = int(detection[2] * frameWidth)
            height = int(detection[3] * frameHeight)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            classIds.append(classId)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])
            objects.append({"id": classId, "box": [left, top, width, height], "confidence": confidence})
    return objects


def getSkeleton(image, threshold=0.1):
    H, W = image.shape[2], image.shape[3]
    keypoints = []
    for i in range(image.shape[1]):
        if i >= 25:
            break
        prob_map = image[0, i, :, :]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        y = point[0] / W
        x = point[1] / H
        if prob > threshold:
            keypoints.append([x, y])
        else:
            keypoints.append([np.nan, np.nan])
    return keypoints


POSE_PARTS = {
    0:  "Nose",
    1:  "Neck",
    2:  "RShoulder",
    3:  "RElbow",
    4:  "RWrist",
    5:  "LShoulder",
    6:  "LElbow",
    7:  "LWrist",
    8:  "MidHip",
    9:  "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle",
    15: "REye",
    16: "LEye",
    17: "REar",
    18: "LEar",
    19: "LBigToe",
    20: "LSmallToe",
    21: "LHeel",
    22: "RBigToe",
    23: "RSmallToe",
    24: "RHeel"
}

def indice(dictionary, value):
    for key, val in dictionary.items():
        if lower(val) == lower(value):
            return key
    return None

# Define pairs of connected keypoints (body parts)
POSE_PAIRS = [(0, 1), (0, 15), (0, 16),
              (15, 17), (16, 18),
              (1, 2), (1, 5), (1, 8),
              (2, 3), (3, 4),
              (5, 6), (6, 7),
              (8, 9), (8, 12),
              (9, 10), (10, 11), (11, 22), (11, 23), (11, 24),
              (12, 13), (13, 14), (14, 19), (14, 20), (14, 21)]
