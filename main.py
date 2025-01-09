import os

import cv2
import numpy as np

from PoseEstimation import *

def main():
    EPSILON = 1
    video_path = 0 # "../ImageProcessing/BodyVideos/body9.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = FPSCounter()

    params = SkeletonTrackerParameters()
    params.use_yolo = True
    params.use_body = True
    params.max_bodies = 1
    params.use_hands = True
    params.use_face = True
    params.hand_skip_frames = 0
    params.models_paths = "PoseEstimation/models"
    tracking = SkeletonTracker(params)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (800, 600))
        img = cv2.flip(img, 1)

        tracking.update(img)

        for detection in tracking.objects_detected:
            detection.displayBox(img)
            detection.displayCenter(img)

        for person in tracking.persons:
            img = person.displaySkeleton(img)

        for aruco in tracking.arucos:
            img = aruco.display(img)

        for hand in tracking.hands:
            img = hand.displaySkeleton(img)

        for face in tracking.faces:
            img = face.displaySkeleton(img)


        fps.update()
        img = fps.display(img)
        cv2.imshow("Resultat", img)
        key = cv2.waitKey(EPSILON) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()