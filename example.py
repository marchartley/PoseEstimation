import cv2
import numpy as np

from FPSCounter import FPSCounter
import Parsers
from Parsers import parseYoloResults, getSkeleton

from SkeletonTracker import SkeletonTracker

def main():
    EPSILON = 1
    video_path = 0 # "../ImageProcessing/BodyVideos/body9.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = FPSCounter()

    tracking = SkeletonTracker(use_yolo=True, use_body=True, max_bodies=1)

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


        fps.update()
        img = fps.display(img)
        cv2.imshow("Resultat", img)
        key = cv2.waitKey(EPSILON) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()