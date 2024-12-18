import time
import random
from typing import Tuple, List

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from Parsers import *

class Person:
    maxId = 0
    maxTrack = 10
    def __init__(self, pos, id = -1):
        if id < 0:
            self.id = Person.maxId + 1
        else:
            self.id = id
        Person.maxId = max(Person.maxId, self.id)

        self.timeAlive = Person.maxTrack
        self.positions = np.array([pos])

        self.skeleton_history = np.zeros((0, 25, 2))
        self.lastSkeleton = np.ones((25, 2)) * np.nan
        self.cachedSkeleton = np.ones((25, 2)) * np.nan
        self.canUseCache = False

        self.face_history = np.zeros((0, 68, 2))
        self.totalTimeAlive = 1

    def update(self):
        if not self.alive():
            self.positions = np.array([None, None])
            self.id = -1
        self.timeAlive -= 1
        self.lastSkeleton = np.ones_like(self.lastSkeleton) * np.nan
        self.totalTimeAlive += 1
        return self

    def refresh(self):
        self.timeAlive = Person.maxTrack

    def alive(self):
        return self.timeAlive >= 0

    def newPos(self, newPos):
        self.positions = np.append(self.positions, [newPos], axis=0)
        if len(self.positions) > Person.maxTrack:
            self.positions = np.delete(self.positions, 0, axis=0)

    def pos(self):
        s = self.skeleton()
        if not np.isnan(s[0]).any():
            return s[0]
        return np.mean(self.positions, axis = 0)

    def newSkeletonPos(self, partId, newPos):
        self.lastSkeleton[partId] = np.array(newPos)
        self.canUseCache = False

    def skeleton(self):
        if self.canUseCache:
            return self.cachedSkeleton
        if not np.isnan(self.lastSkeleton).all():
            self.skeleton_history = np.append(self.skeleton_history, [self.lastSkeleton], axis=0)
        n = 5
        num_articulations = self.skeleton_history.shape[1]  # number of articulations
        num_coordinates = self.skeleton_history.shape[2]  # dimensions of coordinates (e.g., x, y, z)

        # Initialize the array to store the results
        means = np.empty((num_articulations, num_coordinates))

        # Iterate through each articulation
        for articulation in range(num_articulations):
            # Iterate through each coordinate dimension
            for coord in range(num_coordinates):
                # Extract the current articulation's coordinate series
                series = self.skeleton_history[:, articulation, coord]

                # Filter out NaN values
                valid_values = series[~np.isnan(series)]

                # Take the last n values
                last_n_values = valid_values[-n:]

                # Calculate the mean
                means[articulation, coord] = (np.median(last_n_values) + np.median(last_n_values)) * 0.5
        self.cachedSkeleton = means
        self.canUseCache = True
        return self.cachedSkeleton
        # if not np.isnan(self.lastSkeleton).all():
        #     self.skeleton_history = np.append(self.skeleton_history, [self.lastSkeleton], axis=0)
        #     # self.lastSkeleton *= np.nan
        # if len(self.skeleton_history) > Person.maxTrack:
        #     self.skeleton_history = np.delete(self.skeleton_history, 0, axis=0)
        # returnedSkeleton = np.nanmean(self.skeleton_history, axis = 0) # (np.nanmedian(self.skeleton_history, axis = 0) + np.nanmean(self.skeleton_history, axis = 0)) / 2.0
        # return returnedSkeleton

    def getArticulation(self, articulationName):
        iPart = indice(POSE_PARTS, articulationName)
        if iPart is not None:
            return self.skeleton()[iPart]
        return np.array([np.nan, np.nan])

    def articulationVisible(self, articulationName):
        iPart = indice(POSE_PARTS, articulationName)
        if iPart is not None:
            return not np.isnan(self.skeleton()[iPart]).any()
        return False

    def newFace(self, facialLandmarks):
        self.face_history = np.append(self.face_history, [facialLandmarks], axis=0)
        if len(self.face_history) > Person.maxTrack:
            self.face_history = np.delete(self.face_history, 0, axis=0)

    def face(self):
        return np.nanmean(self.face_history, axis = 0)

    def displaySkeleton(self, image):
        pos = self.pos()
        cv2.circle(image, np.array([pos[1] * image.shape[1], pos[0] * image.shape[0]]).astype(int), 5, (0, 0, 255),
                   -1)
        cv2.putText(image, f"{self.id}", np.array([pos[1] * image.shape[1], pos[0] * image.shape[0]]).astype(int),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        skeleton = self.skeleton()
        for pair in POSE_PAIRS:
            bodyA, bodyB = skeleton[pair[0]], skeleton[pair[1]]
            if not np.isnan(bodyA).any() and not np.isnan(bodyB).any():
                cv2.line(image, np.array([bodyA[1] * image.shape[1], bodyA[0] * image.shape[0]]).astype(int),
                         np.array([bodyB[1] * image.shape[1], bodyB[0] * image.shape[0]]).astype(int), (0, 255, 0), 2)

        for iPart in range(25):
            body = skeleton[iPart]
            if not np.isnan(body).any():
                cv2.circle(image, np.array([body[1] * image.shape[1], body[0] * image.shape[0]]).astype(int), 3,
                           (0, 255, 255), -1)

        face = self.face()
        for body in face:
            if not np.isnan(body).any():
                cv2.circle(image, np.array([body[1] * image.shape[1], body[0] * image.shape[0]]).astype(int), 3,
                           (0, 255, 255), -1)
        return image

    def __repr__(self):
        return f"{self.id} ({self.positions[-1]})"


class ObjectDetected:
    def __init__(self, idClasse, label, box, confidence):
        self.idClasse = idClasse
        self.label = label
        self.box = box
        self.confidence = confidence

    def afficherTerminal(self):
        print(f"{self.label} detecté à {int(self.confidence * 100)}% aux coordonnées {self.box}")

    def center(self):
        return np.array(self.box[:2]) + np.array(self.box[2:]) * 0.5

    def distanceToCenter(self, pos_x, pos_y):
        return np.linalg.norm(np.array([pos_x, pos_y]) - self.center())


class SkeletonTracker:
    def __init__(self, use_body = False, max_bodies = 1, use_face = False, pose_resolution = [256, 256], use_yolo = False, yolo_resolution = [256, 256], yolo_threshold = 0.1, use_hands = False, hand_resolution = [256, 256]):
        self.cascade_front_path = "models/haarcascades/haarcascade_frontalface_default.xml"
        self.cascade_smile_path = "models/haarcascades/haarcascade_smile.xml"
        # self.cascade_front_path = "models/haarcascades/haarcascade_frontalface_alt_tree.xml"
        # self.cascade_front_path = "models/haarcascades/haarcascade_frontalface_alt2.xml"
        # self.cascade_front_path = "models/face/haarcascade_frontalface_alt.xml"
        self.fullbody_path = "models/haarcascades/haarcascade_fullbody.xml"
        self.upperbody_path = "models/haarcascades/haarcascade_upperbody.xml"
        self.cascade_profile_path = "models/haarcascades/haarcascade_profileface.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_front_path)
        self.face_profile_cascade = cv2.CascadeClassifier(self.cascade_profile_path)
        self.fullbody_cascade = cv2.CascadeClassifier(self.fullbody_path)
        self.upperbody_cascade = cv2.CascadeClassifier(self.upperbody_path)
        self.smile_cascade = cv2.CascadeClassifier(self.cascade_smile_path)

        self.landmark_detector  = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel("models/face_landmarks/lbfmodel.yaml")

        pose_protoFile_path = "models/pose/body_25/pose_deploy.prototxt"
        pose_weightsFile_path = "models/pose/body_25/pose_iter_584000.caffemodel"
        self.pose_net = cv2.dnn.readNetFromCaffe(pose_protoFile_path, pose_weightsFile_path)
        self.pose_netInputSize = np.array(pose_resolution)
        pose_netOutputSize = np.ceil(self.pose_netInputSize / 8).astype(int)
        self.pose_heatmaps = np.zeros((25, pose_netOutputSize[0], pose_netOutputSize[1]))
        self.max_bodies = max_bodies

        hand_protoFile_path = "models/hand/pose_deploy.prototxt"
        hand_weightsFile_path = "models/hand/pose_iter_120000.caffemodel"
        self.hand_net = cv2.dnn.readNetFromCaffe(hand_protoFile_path, hand_weightsFile_path)
        self.hand_netInputSize = np.array(hand_resolution)
        hand_netOutputSize = np.ceil(self.hand_netInputSize / 8).astype(int)
        self.hand_heatmaps = np.zeros((22, hand_netOutputSize[0], hand_netOutputSize[1]))

        yolo_config_path = "models/yolo/yolov4-tiny.cfg"
        yolo_weights_path = "models/yolo/yolov4-tiny.weights"
        self.yolo_net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
        self.yolo_netInputSize = yolo_resolution
        self.yolo_threshold = yolo_threshold

        classesFile = "models/yolo/coco.names"
        self.yolo_classes = None
        with open(classesFile, 'rt') as f:
            self.yolo_classes = f.read().rstrip('\n').split('\n')

        if self.yolo_net.empty():
            print("Echec")
        else:
            print("Net pret")

        if self.pose_net.empty():
            print("Failed to load the network.")
        else:
            print("Network successfully loaded.")

        self.persons: List[Person] = []
        self.objects_detected = []

        self.use_body = use_body
        self.use_face = use_face
        self.use_yolo = use_yolo
        self.use_hands = use_hands

        self.skip_frames = 2

        self.current_frame = 0

        self.flip = True

    def update(self, img: np.ndarray):

        if self.use_yolo:
            inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, self.yolo_netInputSize, (127.5, 127.5, 127.5), crop=False, swapRB=True)
            self.yolo_net.setInput(inpBlob)
            out = self.yolo_net.forward()
            self.objects_detected = self.detections_Yolo(img, out)

        if self.use_body:
            if self.current_frame % (self.skip_frames + 1) == 0:
                self.getPoses(img, 0.1)
                self.persons = sorted([p for p in self.persons], key = lambda p: -p.totalTimeAlive)[:self.max_bodies]

        if self.use_hands:
            if self.current_frame % (self.skip_frames + 1) == 0:
                self.getHands(img, 0.1)

        if self.use_face:
            self.getFaces(img)

        self.current_frame += 1

    def getFaces(self, img):
        imH, imW, imC = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        for person in self.persons:
            if not person.alive():
                continue
            skeleton = person.skeleton()
            nose = skeleton[0]
            neck = skeleton[1]
            lEar = skeleton[18]
            rEar = skeleton[17]
            neckToNose = nose - neck
            leftToRight = rEar - lEar
            topLeft = nose + neckToNose + leftToRight
            topRight = nose + neckToNose - leftToRight
            bottomLeft = nose - neckToNose + leftToRight
            bottomRight = nose - neckToNose - leftToRight

            x_min = int(min(max(0.0, min(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1])), 1.0) * imW)
            y_min = int(min(max(0.0, min(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0])), 1.0) * imH)
            x_max = int(min(max(0.0, max(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1])), 1.0) * imW)
            y_max = int(min(max(0.0, max(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0])), 1.0) * imH)
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min

            cropped = gray[y:(y+h), x:(x+w)]
            detection = self.face_cascade.detectMultiScale(cropped, scaleFactor=1.1, minNeighbors=0, minSize=(h // 2, w // 2), maxSize=(h * 2, w * 2))
            if len(detection) == 0:
                detection = self.face_profile_cascade.detectMultiScale(cropped, scaleFactor=1.1, minNeighbors=0,
                                                               minSize=(h // 2, w // 2), maxSize=(h * 2, w * 2))
            if len(detection) > 0:
                _, landmarks = self.landmark_detector.fit(img, np.array([np.array(detection[0]) + np.array([x, y, 0, 0])]))
                if len(landmarks) > 0:
                    person.newFace([[m[1], m[0]] for m in landmarks[0][0] / np.array([imW, imH])])
            elif len(person.face_history) > 0:
                person.newFace(np.ones_like(person.face_history[0]) * np.nan)


    def get_keypoints(self, image, threshold=0.1):
        H, W = image.shape[2], image.shape[3]
        keypoints = []
        for i in range(image.shape[1]):
            if i >= 25:
                break
            prob_map = image[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            x = (image.shape[1] * point[0]) / W
            y = (image.shape[0] * point[1]) / H
            if prob > threshold:
                keypoints.append([int(x), int(y)])
            else:
                keypoints.append([None, None])
        return keypoints

    def extract_keypoints_from_heatmap(self, heatmap, threshold=0.1):
        keypoints = []
        heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
        heatmap[heatmap < threshold] = 0
        # heatmap[heatmap >= threshold] = 1
        peaks = (cv2.dilate(heatmap, kernel=np.ones((3, 3))) == heatmap) * (
                    heatmap > threshold)  # cv2.dilate(heatmap, kernel=np.ones((10, 10))) == heatmap
        peaks_coords = np.argwhere(peaks)
        for y, x in peaks_coords:
            if heatmap[y, x] >= threshold:
                keypoints.append((int(x), int(y), heatmap[y, x]))
        return keypoints

    def getPoses(self, img, threshold=0.1):
        image = img.copy()
        if self.flip:
            image = cv2.flip(image, 1)
        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, self.pose_netInputSize, (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.pose_net.setInput(inpBlob)
        output = self.pose_net.forward()

        output = output[0, :25]
        decay = 0.9
        big = np.array([cv2.resize(out, (self.pose_heatmaps[0].shape[1], self.pose_heatmaps[0].shape[0]), interpolation=cv2.INTER_CUBIC) for out in output])
        
        self.pose_heatmaps = self.pose_heatmaps * (1 - decay) + big * decay
        self.pose_heatmaps[self.pose_heatmaps < threshold] = 0

        heatmap = output[0]
        heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
        heatmap[heatmap < threshold] = 0
        # heatmap[heatmap >= threshold] = 1
        peaks = (cv2.dilate(heatmap, kernel=np.ones((3, 3))) == heatmap) * (
                heatmap > threshold)

        peaks_coords = (np.argwhere(peaks))
        peaks_vals = [heatmap[x, y] for x, y in peaks_coords]
        peaks_coords = peaks_coords[np.argsort(-heatmap[tuple(peaks_coords.T)])].astype(np.float32) / np.array([heatmap.shape[0], heatmap.shape[1]])

        #
        # if self.max_bodies == 1:
        #     keypoints = self.get_keypoints(output, threshold)
        #     if keypoints:
        #         if len(self.persons) == 0:
        #             self.persons.append(Person(keypoints[0], 1))
        #         elif len(self.persons) > 1:
        #             self.persons = [self.persons[0]]
        #         self.persons[0].newPos(keypoints[0])
        #         for iPart in POSE_PARTS:
        #             if keypoints[iPart][0] is not None:
        #                 self.persons[0].newSkeletonPos(iPart, np.array(keypoints[iPart]) / (self.pose_netInputSize / 8))
        #     return

        self.persons = [pers.update() for pers in self.persons if pers.alive()]
        # if self.max_bodies == 1:
        self.persons = self.persons[:self.max_bodies]

        distance_threshold = 0.25 ** 2
        costMatrix = np.ones((len(self.persons), len(peaks_coords))) * np.inf
        for iPerson, person in enumerate(self.persons):
            pos = person.pos()
            for iCoord, coord in enumerate(peaks_coords):
                diff = (coord - pos) ** 2
                sqrDist = diff[0] + diff[1]
                if sqrDist < distance_threshold:
                    costMatrix[iPerson, iCoord] = sqrDist / peaks_vals[iCoord]

        newPersonsCoords = [i for i in range(len(peaks_coords))]
        try:
            row_ind, col_ind = linear_sum_assignment(costMatrix)
            for iPerson, iCoord in zip(row_ind, col_ind):
                self.persons[iPerson].newPos(peaks_coords[iCoord])
                self.persons[iPerson].refresh()
                newPersonsCoords.remove(iCoord)

            for iCoord in newPersonsCoords:
                self.persons.append(Person(peaks_coords[iCoord]))
        except:
            pass

        for iPart in range(self.pose_heatmaps.shape[0]):
            heatmap = output[iPart]
            heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
            heatmap[heatmap < threshold] = 0
            peaks = (cv2.dilate(heatmap, kernel=np.ones((3, 3))) == heatmap) * (
                    heatmap > threshold)
            peaks_coords = np.argwhere(peaks)
            peaks_vals = [heatmap[x, y] for x, y in peaks_coords]
            peaks_coords = peaks_coords[np.argsort(-heatmap[tuple(peaks_coords.T)])].astype(np.float32) / np.array([heatmap.shape[0], heatmap.shape[1]])

            costMatrix = np.ones((len(self.persons), len(peaks_coords))) * np.inf
            for iPerson, person in enumerate(self.persons):
                skeleton = person.skeleton()
                pos = skeleton[iPart]
                otherParts = [skeleton[A] if A != iPart else skeleton[B] for A, B in POSE_PAIRS if iPart in (A, B) and not np.isnan(skeleton[A]).any() and not np.isnan(skeleton[B]).any()]
                if np.isnan(pos).any():
                    pos = person.pos()
                for iCoord, coord in enumerate(peaks_coords):
                    diff = (coord - pos) ** 2
                    for other in otherParts:
                        diff += (coord - other)**2
                    sqrDist = diff[0] + diff[1]
                    costMatrix[iPerson, iCoord] = sqrDist / peaks_vals[iCoord]

            try:
                row_ind, col_ind = linear_sum_assignment(costMatrix)

                for iPerson, iCoord in zip(row_ind, col_ind):
                    self.persons[iPerson].newSkeletonPos(iPart, peaks_coords[iCoord])
            except Exception as e:
                for person in self.persons:
                    person.newSkeletonPos(iPart, [None, None])

    def getHands(self, image, threshold=0.1):
        inpBlob = cv2.dnn.blobFromImage(image, 1.0, self.hand_netInputSize, (127.5, 127.5, 127.5), swapRB=False, crop=False)
        self.hand_net.setInput(inpBlob)
        output = self.hand_net.forward()

        output = output[0, :]

        decay = 1.0
        big = np.array([cv2.resize(out, (self.hand_heatmaps[0].shape[1], self.hand_heatmaps[0].shape[0])) for out in output])
        self.hand_heatmaps = self.hand_heatmaps * (1 - decay) + big * decay
        self.hand_heatmaps[self.hand_heatmaps < threshold] = 0

        heatmap = output[0]
        self.hand_heatmaps[-1] *= 0
        for i in range(22):
            self.hand_heatmaps[-1] = np.maximum(self.hand_heatmaps[-1], output[i])
        # heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
        # heatmap[heatmap < threshold] = 0
        # # heatmap[heatmap >= threshold] = 1
        # peaks = (cv2.dilate(heatmap, kernel=np.ones((3, 3))) == heatmap) * (
        #         heatmap > threshold)  # cv2.dilate(heatmap, kernel=np.ones((10, 10))) == heatmap
        # peaks_coords = np.argwhere(peaks) / np.array([heatmap.shape[0], heatmap.shape[1]])
        #
        # self.persons = [pers.update() for pers in self.persons if pers.alive()]
        #
        # distance_threshold = 0.25 ** 2
        # costMatrix = np.ones((len(self.persons), len(peaks_coords))) * np.inf
        # for iPerson, person in enumerate(self.persons):
        #     pos = person.pos()
        #     for iCoord, coord in enumerate(peaks_coords):
        #         diff = (coord - pos) ** 2
        #         sqrDist = diff[0] + diff[1]
        #         if sqrDist < distance_threshold:
        #             costMatrix[iPerson, iCoord] = sqrDist
        #
        # newPersonsCoords = [i for i in range(len(peaks_coords))]
        # try:
        #     row_ind, col_ind = linear_sum_assignment(costMatrix)
        #     for iPerson, iCoord in zip(row_ind, col_ind):
        #         self.persons[iPerson].newPos(peaks_coords[iCoord])
        #         self.persons[iPerson].refresh()
        #         newPersonsCoords.remove(iCoord)
        #
        #     for iCoord in newPersonsCoords:
        #         self.persons.append(Person(peaks_coords[iCoord]))
        # except:
        #     pass
        #
        # for iPart in range(self.hand_heatmaps.shape[0]):
        #     heatmap = output[iPart]
        #     heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
        #     heatmap[heatmap < threshold] = 0
        #     peaks = (cv2.dilate(heatmap, kernel=np.ones((3, 3))) == heatmap) * (
        #             heatmap > threshold)
        #     peaks_coords = np.argwhere(peaks) / np.array([heatmap.shape[0], heatmap.shape[1]])
        #
        #     costMatrix = np.ones((len(self.persons), len(peaks_coords))) * np.inf
        #     for iPerson, person in enumerate(self.persons):
        #         pos = person.pos()
        #         for iCoord, coord in enumerate(peaks_coords):
        #             diff = (coord - pos) ** 2
        #             sqrDist = diff[0] + diff[1]
        #             costMatrix[iPerson, iCoord] = sqrDist
        #
        #     try:
        #         row_ind, col_ind = linear_sum_assignment(costMatrix)
        #
        #         for iPerson, iCoord in zip(row_ind, col_ind):
        #             self.persons[iPerson].newSkeletonPos(iPart, peaks_coords[iCoord])
        #     except Exception as e:
        #         for person in self.persons:
        #             person.newSkeletonPos(iPart, [None, None])


    def detections_Yolo(self, frame, outs):
        res = parseYoloResults(frame, outs, self.yolo_threshold)

        objects: List[ObjectDetected] = []
        for detection in res:
            classId = detection["id"]
            label = self.yolo_classes[classId]
            box = detection["box"]
            confidence = detection["confidence"]
            objDetection = ObjectDetected(classId, label, box, confidence)
            needToBeAdded = True
            for others in objects:
                if others.idClasse == classId and others.distanceToCenter(*objDetection.center()) < 0.5 * (box[2] ** 2 + box[3] ** 2) ** 0.5:
                    needToBeAdded = False
                    break
            if needToBeAdded:
                objects.append(objDetection)
        return objects
