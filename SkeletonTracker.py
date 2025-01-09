import os.path
from typing import Tuple, List

import cv2
from scipy.optimize import linear_sum_assignment

from .Parsers import *

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics import YOLO

class Detection:
    maxTrack = 10
    def __init__(self):
        self.timeAlive = Detection.maxTrack
        self.totalTimeAlive = 1

    def update(self):
        self.timeAlive -= 1
        self.totalTimeAlive += 1
        return self

    def alive(self):
        return self.timeAlive > 0

    def refresh(self):
        self.timeAlive = Detection.maxTrack

class SkeletonDetection(Detection):
    maxId = 0
    def __init__(self, nb_articulations = 0, id = -1, vertices_names=None, mesh_edges=None):
        super().__init__()
        if mesh_edges is None:
            mesh_edges = []
        if vertices_names is None:
            vertices_names = {}
        if id < 0:
            self.id = SkeletonDetection.maxId + 1
        else:
            self.id = id
        SkeletonDetection.maxId = max(SkeletonDetection.maxId, self.id)
        self.nb_articulations = nb_articulations
        self.skeleton_history = np.zeros((0, self.nb_articulations, 2))
        self.lastSkeleton = np.ones((self.nb_articulations, 2)) * np.nan
        self.cachedSkeleton = np.ones((self.nb_articulations, 2)) * np.nan
        self.canUseCache = False

        self.mesh_edges = mesh_edges
        self.vertices_names = vertices_names

    def update(self):
        super().update()
        if not self.alive():
            self.id = -1
        self.lastSkeleton = np.ones_like(self.lastSkeleton) * np.nan
        return self

    def newSkeleton(self, newPose):
        self.lastSkeleton = np.array(newPose)
        if len(self.lastSkeleton) != len(self.cachedSkeleton):
            self.lastSkeleton = np.concatenate([self.lastSkeleton, np.array([[np.nan, np.nan]] * int(len(self.cachedSkeleton) - len(self.lastSkeleton)))])
        self.canUseCache = False

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

    def getArticulation(self, articulationName):
        iPart = indice(self.vertices_names, articulationName)
        if iPart is not None:
            return self.skeleton()[iPart]
        return np.array([np.nan, np.nan])

    def articulationVisible(self, articulationName):
        iPart = indice(self.vertices_names, articulationName)
        if iPart is not None:
            return not np.isnan(self.skeleton()[iPart]).any()
        return False

    def displaySkeleton(self, image: np.ndarray, color = (0, 255, 255)):
        skeleton = self.skeleton()
        for pair in self.mesh_edges:
            bodyA, bodyB = skeleton[pair[0]], skeleton[pair[1]]
            if not np.isnan(bodyA).any() and not np.isnan(bodyB).any():
                cv2.line(image, np.array([bodyA[1] * image.shape[1], bodyA[0] * image.shape[0]]).astype(int),
                         np.array([bodyB[1] * image.shape[1], bodyB[0] * image.shape[0]]).astype(int), color, 2)

        for iPart in range(self.nb_articulations):
            body = skeleton[iPart]
            if not np.isnan(body).any():
                cv2.circle(image, np.array([body[1] * image.shape[1], body[0] * image.shape[0]]).astype(int), 3,
                           color, -1)
        return image



class Person(SkeletonDetection):
    def __init__(self, pos, id = -1):
        super().__init__(25, id, POSE_PARTS, POSE_PAIRS)
        self.positions = np.array([pos])
    #
    # def skeleton(self):
    #     if self.canUseCache:
    #         return self.cachedSkeleton
    #     if not np.isnan(self.lastSkeleton).all():
    #         self.skeleton_history = np.append(self.skeleton_history, [self.lastSkeleton], axis=0)
    #     n = 5
    #     num_articulations = self.skeleton_history.shape[1]  # number of articulations
    #     num_coordinates = self.skeleton_history.shape[2]  # dimensions of coordinates (e.g., x, y, z)
    #
    #     # Initialize the array to store the results
    #     means = np.empty((num_articulations, num_coordinates))
    #
    #     # Iterate through each articulation
    #     for articulation in range(num_articulations):
    #         # Iterate through each coordinate dimension
    #         for coord in range(num_coordinates):
    #             # Extract the current articulation's coordinate series
    #             series = self.skeleton_history[:, articulation, coord]
    #
    #             # Filter out NaN values
    #             valid_values = series[~np.isnan(series)]
    #
    #             # Take the last n values
    #             last_n_values = valid_values[-n:]
    #
    #             # Calculate the mean
    #             means[articulation, coord] = (np.median(last_n_values) + np.median(last_n_values)) * 0.5
    #     self.cachedSkeleton = means
    #     self.canUseCache = True
    #     return self.cachedSkeleton
    #
    # def getArticulation(self, articulationName):
    #     iPart = indice(POSE_PARTS, articulationName)
    #     if iPart is not None:
    #         return self.skeleton()[iPart]
    #     return np.array([np.nan, np.nan])
    #
    # def articulationVisible(self, articulationName):
    #     iPart = indice(POSE_PARTS, articulationName)
    #     if iPart is not None:
    #         return not np.isnan(self.skeleton()[iPart]).any()
    #     return False
    #
    def newPos(self, newPos):
        self.positions = np.append(self.positions, [newPos], axis=0)
        if len(self.positions) > Person.maxTrack:
            self.positions = np.delete(self.positions, 0, axis=0)

    def pos(self):
        s = self.skeleton()
        if not np.isnan(s[0]).any():
            return s[0]
        return np.mean(self.positions, axis = 0)
    #
    # def displaySkeleton(self, image, color = (0, 255, 255)):
    #     # pos = self.pos()
    #     # # cv2.circle(image, np.array([pos[1] * image.shape[1], pos[0] * image.shape[0]]).astype(int), 5, (0, 0, 255),
    #     # #            -1)
    #     # cv2.putText(image, f"{self.id}", np.array([pos[1] * image.shape[1], pos[0] * image.shape[0]]).astype(int),
    #     #             cv2.FONT_HERSHEY_SIMPLEX, 1,
    #     #             color, 2, cv2.LINE_AA)
    #
    #     skeleton = self.skeleton()
    #     for pair in POSE_PAIRS:
    #         bodyA, bodyB = skeleton[pair[0]], skeleton[pair[1]]
    #         if not np.isnan(bodyA).any() and not np.isnan(bodyB).any():
    #             posA = np.array([bodyA[1] * image.shape[1], bodyA[0] * image.shape[0]]).astype(int)
    #             posB = np.array([bodyB[1] * image.shape[1], bodyB[0] * image.shape[0]]).astype(int)
    #             cv2.line(image, posA, posB, color, 2)
    #
    #     for iPart in range(25):
    #         body = skeleton[iPart]
    #         if not np.isnan(body).any():
    #             pos = np.array([body[1] * image.shape[1], body[0] * image.shape[0]]).astype(int)
    #             cv2.putText(image, f"{iPart}", pos, cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                         color, 2, cv2.LINE_AA)
    #             cv2.circle(image, pos, 3,
    #                        (0, 255, 255), -1)
    #
    #     return image



class HandDetected(SkeletonDetection):
    def __init__(self, id = -1):
        super().__init__(22, id, HAND_PARTS, HAND_PAIRS)
    #
    # def getArticulation(self, articulationName):
    #     iPart = indice(HAND_PARTS, articulationName)
    #     if iPart is not None:
    #         return self.skeleton()[iPart]
    #     return np.array([np.nan, np.nan])
    #
    # def articulationVisible(self, articulationName):
    #     iPart = indice(HAND_PARTS, articulationName)
    #     if iPart is not None:
    #         return not np.isnan(self.skeleton()[iPart]).any()
    #     return False
    #
    # def displaySkeleton(self, image, color = (0, 255, 255)):
    #     skeleton = self.skeleton()
    #     for iPart in range(self.nb_articulations):
    #         body = skeleton[iPart]
    #         if not np.isnan(body).any():
    #             position = np.array([body[1] * image.shape[1], body[0] * image.shape[0]]).astype(int)
    #             cv2.circle(image, position, 3, color, -1)
    #             cv2.putText(image, HAND_PARTS[iPart], position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
    #
    #     for pair in HAND_PAIRS:
    #         bodyA, bodyB = skeleton[pair[0]], skeleton[pair[1]]
    #         if not np.isnan(bodyA).any() and not np.isnan(bodyB).any():
    #             posA = np.array([bodyA[1] * image.shape[1], bodyA[0] * image.shape[0]]).astype(int)
    #             posB = np.array([bodyB[1] * image.shape[1], bodyB[0] * image.shape[0]]).astype(int)
    #             cv2.line(image, posA, posB, color, 2)
    #     return image


class FaceDetected(SkeletonDetection):
    def __init__(self, id = -1):
        super().__init__(479, id, [], FACEMESH_TESSELATION)

    def get_center(self):
        return np.nanmean(self.skeleton(), axis = 0)





class ObjectDetected(Detection):
    def __init__(self, idClasse, label, box, confidence):
        super().__init__()
        self.idClasse = idClasse
        self.label = label
        self.box = np.array(box, dtype=int)
        self.confidence = confidence

    def afficherTerminal(self):
        print(f"{self.label} detecté à {int(self.confidence * 100)}% aux coordonnées {self.box}")

    def displayBox(self, img, color = (0, 0, 255)):
        box = self.box
        cv2.rectangle(img, [box[0], box[1]], [box[0] + box[2], box[1] + box[3]], color, 2)
        cv2.putText(img,
                    f"{self.label} ({int(self.confidence * 100)}%)",
                    (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)
        return img
    def displayCenter(self, img, color = (0, 0, 255)):
        position = self.center()
        label = self.label
        cv2.circle(img, position.astype(int), 3, color, -1)
        cv2.putText(img, f"{label}",
        position.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

    def center(self):
        return np.array(self.box[:2]) + np.array(self.box[2:]) * 0.5

    def distanceToCenter(self, pos_x, pos_y):
        return np.linalg.norm(np.array([pos_x, pos_y]) - self.center())


class Aruco(Detection):
    def __init__(self, arucoID, corners):
        super().__init__()
        self.aruco_id = arucoID
        self.corners = corners
        self.previous_corners = np.array(corners)

    def topLeft(self):
        dt = Aruco.maxTrack - self.timeAlive
        return (self.corners[0] - self.previous_corners[0]) * dt + self.corners[0]
    def topRight(self):
        dt = Aruco.maxTrack - self.timeAlive
        return (self.corners[1] - self.previous_corners[1]) * dt + self.corners[1]
    def bottomRight(self):
        dt = Aruco.maxTrack - self.timeAlive
        return (self.corners[2] - self.previous_corners[2]) * dt + self.corners[2]
    def bottomLeft(self):
        dt = Aruco.maxTrack - self.timeAlive
        return (self.corners[3] - self.previous_corners[3]) * dt + self.corners[3]

    def addPosition(self, newCorners):
        self.previous_corners = np.array(self.corners)
        self.corners = np.array(newCorners)
        self.refresh()

    def center(self):
        return (self.topLeft() + self.topRight() + self.bottomRight() + self.topLeft()) / 4

    def display(self, img, color = (0, 255, 0)):
        cv2.putText(img, f"{self.aruco_id}", self.center().astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.polylines(img, [np.array([self.topRight(), self.topLeft(), self.bottomLeft(), self.bottomRight()]).astype(int)], True, color, 2)
        return img


class SkeletonTrackerParameters:
    def __init__(self):
        root_path = os.path.dirname(__file__)
        self.models_paths = root_path + "/models"

        self.use_body = False
        self.max_bodies = 1
        self.body_model_filename = "pose_landmarker.task"
        self.body_path = self.models_paths + "/pose/mediapipe/" + self.body_model_filename
        self.pose_skip_frames = 2

        self.use_face = False
        self.face_model_filename = "face_landmarker_v2_with_blendshapes.task"
        self.face_landmarks_path = self.models_paths + "/face_landmarks/" + self.face_model_filename
        self.face_skip_frames = 2

        self.use_hands = False
        self.hand_model_filename = "hand_landmarker.task"
        self.hand_model_path = self.models_paths + "/hand/" + self.hand_model_filename
        self.hand_skip_frames = 3

        self.use_yolo = False
        self.yolo_threshold = 0.1
        self.yolo_model = "yolo11m"
        self.yolo_model_path = self.models_paths + "/yolo/" + self.yolo_model + ".pt"
        self.yolo_skip_frames = 1

        self.use_aruco = False
        self.aruco_dictionary = cv2.aruco.DICT_6X6_1000
        self.aruco_skip_frames = 0

        self.flip_image = False

        self.init_paths()

    def init_paths(self):
        self.body_path = self.models_paths + "/pose/mediapipe/" + self.body_model_filename
        self.face_landmarks_path = self.models_paths + "/face_landmarks/" + self.face_model_filename
        self.hand_model_path = self.models_paths + "/hand/" + self.hand_model_filename
        self.yolo_model_path = self.models_paths + "/yolo/" + self.yolo_model + ".pt"

    def set_models_folder_path(self, folder_path: str):
        self.models_paths = folder_path
        self.init_paths()


class SkeletonTracker:
    def __init__(self, parameters: SkeletonTrackerParameters = None, **kwargs):
        if parameters is None:
            parameters = SkeletonTrackerParameters()
        for key, val in kwargs.items():
            if key not in parameters.__dict__:
                print(f"The parameter '{key}' is not taken into account in this version")
            parameters.__dict__[key] = val


        self.use_body = parameters.use_body
        self.use_face = parameters.use_face
        self.use_yolo = parameters.use_yolo
        self.use_hands = parameters.use_hands
        self.use_aruco = parameters.use_aruco

        self.pose_net = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=parameters.body_path),
            output_segmentation_masks=True))
        self.max_bodies = parameters.max_bodies

        self.face_model = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=parameters.face_landmarks_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=self.max_bodies))

        self.hand_net = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=parameters.hand_model_path),
            num_hands=self.max_bodies * 2))

        self.yolo_net = YOLO(parameters.yolo_model_path)
        self.yolo_threshold = parameters.yolo_threshold

        aruco_dictionary = cv2.aruco.getPredefinedDictionary(parameters.aruco_dictionary)
        aruco_parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
        self.aruco_detector = detector

        self.persons: List[Person] = []
        self.objects_detected: List[ObjectDetected] = []
        self.arucos: List[Aruco] = []
        self.hands: List[HandDetected] = []
        self.faces: List[FaceDetected] = []

        self.current_frame = 0

        self.yolo_skip = parameters.yolo_skip_frames
        self.pose_skip = parameters.pose_skip_frames
        self.face_skip = parameters.face_skip_frames
        self.hand_skip = parameters.hand_skip_frames
        self.aruco_skip = parameters.aruco_skip_frames

        self.flip = parameters.flip_image

    def update(self, img: np.ndarray):

        if self.use_yolo:
            if self.current_frame % (self.yolo_skip + 1) == 0:
                self.getYolo(img, self.yolo_threshold)

        if self.use_body:
            if self.current_frame % (self.pose_skip + 1) == 0:
                self.getPoses(img, 0.1)

        if self.use_hands:
            if self.current_frame % (self.hand_skip + 1) == 0:
                self.getHands(img, -1.0)

        if self.use_aruco:
            if self.current_frame % (self.aruco_skip + 1) == 0:
                self.getAruco(img)

        if self.use_face:
            if self.current_frame % (self.face_skip + 1) == 0:
                self.getFaces(img)

        self.current_frame += 1

    def getYolo(self, img: np.ndarray, threshold: float = 0.1):
        results = self.yolo_net(img, verbose=False)[0]
        self.objects_detected = self.detections_YoloV11(results)
        return self.objects_detected

    def getFaces(self, img):
        image = img.copy()
        if self.flip:
            image = cv2.flip(image, 1)

        face_landmarks = self.face_model.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)))

        self.faces = [h for h in self.faces if h.update().alive()]
        for i, face in enumerate(face_landmarks.face_landmarks):
            pose = np.array([[h.y, h.x] for h in face] + [[np.nan, np.nan]])
            pose_center = np.nanmean(pose, axis = 0)

            foundExisting = False
            for existingFace in self.faces:
                if np.linalg.norm(existingFace.get_center()**2 - pose_center**2) < 20:
                    existingFace.newSkeleton(pose)
                    existingFace.refresh()
                    foundExisting = True
                    break
            if not foundExisting:
                newFace = FaceDetected()
                newFace.newSkeleton(pose)
                self.faces.append(newFace)
        return self.faces

    @staticmethod
    def _mediapipe_body_poses_to_body25(media_poses):
        return np.array([
            media_poses[0],
            (media_poses[11] + media_poses[12]) / 2,
            media_poses[11],
            media_poses[13],
            media_poses[15],
            media_poses[12],
            media_poses[14],
            media_poses[16],
            (media_poses[24] + media_poses[23]) / 2,
            media_poses[23],
            media_poses[25],
            media_poses[27],
            media_poses[24],
            media_poses[26],
            media_poses[28],
            media_poses[2],
            media_poses[5],
            media_poses[7],
            media_poses[8],
            media_poses[30],
            media_poses[32],
            media_poses[32],
            media_poses[29],
            media_poses[31],
            media_poses[31],
            ])
    def getPoses(self, img, threshold=0.1):
        image = img.copy()
        if self.flip:
            image = cv2.flip(image, 1)


        person_landmarks = self.pose_net.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)))
        self.persons = [h for h in self.persons if h.update().alive()]
        for i, person in enumerate(person_landmarks.pose_landmarks):
            pose = SkeletonTracker._mediapipe_body_poses_to_body25(np.array([[h.y, h.x] for i, h in enumerate(person)]))
            idx = 1
            foundExisting = False
            for existing_persons in self.persons:
                if existing_persons.id == idx:
                    existing_persons.newSkeleton(pose)
                    existing_persons.refresh()
                    foundExisting = True
                    break
            if not foundExisting:
                newPerson = Person(idx)
                newPerson.newSkeleton(pose)
                self.persons.append(newPerson)
        return self.persons

    def getHands(self, img, threshold=0.0):
        image = img.copy()
        if self.flip:
            image = cv2.flip(image, 1)

        hand_landmarks = self.hand_net.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)))

        self.hands = [h for h in self.hands if h.update().alive()]
        for i, hand in enumerate(hand_landmarks.hand_landmarks):
            idx = hand_landmarks.handedness[i][0].index
            pose = np.array([[h.y, h.x] for h in hand] + [[np.nan, np.nan]])

            foundExisting = False
            for existing_hands in self.hands:
                if existing_hands.id == idx:
                    existing_hands.newSkeleton(pose)
                    existing_hands.refresh()
                    foundExisting = True
                    break
            if not foundExisting:
                newHand = HandDetected(idx)
                newHand.newSkeleton(pose)
                self.hands.append(newHand)
        return self.hands


    def detections_YoloV11(self, results):
        objects: List[ObjectDetected] = [obj for obj in self.objects_detected if obj.update().alive()]
        for i in range(len(results.boxes)):
            res = results.boxes[i]
            classId = int(res.cls.cpu())
            label = results.names[classId]
            box = res.xywh.cpu().numpy()[0]
            box[0] -= box[2] * 0.5
            box[1] -= box[3] * 0.5
            confidence = float(res.conf.cpu())
            objDetection = ObjectDetected(classId, label, box, confidence)
            needToBeAdded = True
            for others in objects:
                if others.idClasse == classId and others.distanceToCenter(*objDetection.center()) < 0.5 * (
                        box[2] ** 2 + box[3] ** 2) ** 0.5:
                    needToBeAdded = False
                    others.confidence = max(others.confidence, confidence)
                    others.box = np.array(box, dtype=int)
                    others.refresh()
            if needToBeAdded:
                objects.append(objDetection)
        self.objects_detected = objects
        return self.objects_detected


    def detections_Yolo(self, frame, outs):
        res = parseYoloResults(frame, outs, self.yolo_threshold)

        objects: List[ObjectDetected] = [obj for obj in self.objects_detected if obj.update().alive()]
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
                    others.confidence = max(others.confidence, confidence)
                    others.box = box
                    others.refresh()
            if needToBeAdded:
                objects.append(objDetection)
        self.objects_detected = objects
        return self.objects_detected


    def getAruco(self, img):
        self.arucos = [aruco for aruco in self.arucos if aruco.update().alive()]
        markerCorners, markerIds, rejectedCandidates = self.aruco_detector.detectMarkers(img)
        if len(markerCorners) > 0:
            ids = markerIds.flatten()
            for (markerCorner, markerID) in zip(markerCorners, ids):
                corners = markerCorner.reshape((4, 2))
                foundAnExistingAruco = False
                for aruco in self.arucos:
                    if aruco.aruco_id == markerID:
                        aruco.addPosition(corners)
                        foundAnExistingAruco = True
                if not foundAnExistingAruco:
                    self.arucos.append(Aruco(markerID, corners))

        # Apply mirror here
        markerCorners, markerIds, rejectedCandidates = self.aruco_detector.detectMarkers(cv2.flip(img, 1))
        if len(markerCorners) > 0:
            ids = markerIds.flatten()
            for (markerCorner, markerID) in zip(markerCorners, ids):
                corners = markerCorner.reshape((4, 2))
                corners = np.array([[img.shape[1] - c[0], c[1]] for c in corners])
                foundAnExistingAruco = False
                for aruco in self.arucos:
                    if aruco.aruco_id == markerID:
                        aruco.addPosition(corners)
                        foundAnExistingAruco = True
                if not foundAnExistingAruco:
                    self.arucos.append(Aruco(markerID, corners))
        return self.arucos


def example():
    EPSILON = 1
    video_path = 0 # "../ImageProcessing/BodyVideos/body9.mp4"
    cap = cv2.VideoCapture(video_path)

    parameters = SkeletonTrackerParameters()
    parameters.use_body = True
    parameters.max_bodies = 1
    parameters.use_yolo = True
    parameters.use_aruco = True

    tracking = SkeletonTracker(parameters)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (800, 600))
        img = cv2.flip(img, 1)

        tracking.update(img)

        for detection in tracking.objects_detected:
            detection.displayBox(img, (255, 0, 0))

        for person in tracking.persons:
            img = person.displaySkeleton(img, (0, 255, 0))

        for hand in tracking.hands:
            img = hand.displaySkeleton(img, (0, 0, 255))

        for aruco in tracking.arucos:
            img = aruco.display(img, (0, 0, 255))

        cv2.imshow("Resultat", img)
        key = cv2.waitKey(EPSILON) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    example()