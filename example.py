import cv2
import numpy as np

from FPSCounter import FPSCounter
import Parsers
from Parsers import parseYoloResults, getSkeleton

from SkeletonTracker import SkeletonTracker

def main():
    EPSILON = 1
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    fps = FPSCounter()

    tracking = SkeletonTracker(use_yolo=True, use_body=True, max_bodies=2)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (800, 600))
        img = cv2.flip(img, 1)

        tracking.update(img)

        objets_detectes = tracking.objects_detected

        for detection in objets_detectes:
            box = detection.box
            cv2.rectangle(img, [box[0], box[1]], [box[0] + box[2], box[1] + box[3]], (0, 0, 255), 2)
            cv2.putText(img,
                        f"{detection.label} ({int(detection.confidence * 100)}%)",
                        (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        for person in tracking.persons:
            img = person.displaySkeleton(img)

        couleur = (255, 0, 0)
        if len(tracking.persons) > 0:
            if person.articulationVisible("RWrist"):
                position = (person.getArticulation("RWrist") * np.array([600, 800]))
                x = int(position[1])
                y = int(position[0])
                cv2.circle(img, (x, y), 3, couleur, -1)

        for objet_detecte in tracking.objects_detected:
            position = objet_detecte.center()
            label = objet_detecte.label
            cv2.circle(img, position.astype(int), 3, couleur, -1)
            cv2.putText(img, f"{label}",
            position.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, couleur, 2)
        #
        # couleur = (0, 255, 255)
        # for iArticulation, articulation in enumerate(articulations):
        #     if not np.isnan(articulation).any():
        #         position = np.array([articulation[1] * img.shape[1], articulation[0] * img.shape[0]])
        #         nom = Parsers.POSE_PARTS[iArticulation]
        #         cv2.circle(img, position.astype(int), 3, couleur, -1)
        #         cv2.putText(img, f"{nom}", position.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, couleur, 2)
        #
        # for pair in Parsers.POSE_PAIRS:
        #     artA, artB = articulations[pair[0]], articulations[pair[1]]
        #     if not np.isnan(artA).any() and not np.isnan(artB).any():
        #         posA = np.array([artA[1] * img.shape[1], artA[0] * img.shape[0]])
        #         posB = np.array([artB[1] * img.shape[1], artB[0] * img.shape[0]])
        #         cv2.line(img, posA.astype(int), posB.astype(int), couleur, 2)

        fps.update()
        img = fps.display(img)
        cv2.imshow("Resultat", img)
        key = cv2.waitKey(EPSILON) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()
# Bonne pratique : ajouter cet idiome
# dans nos scripts
if __name__ == "__main__":
    main()