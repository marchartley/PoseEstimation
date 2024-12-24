import os.path
from typing import Tuple, List

import cv2
from scipy.optimize import linear_sum_assignment

from PoseEstimation.Parsers import *

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
    def __init__(self, nb_articulations = 0, id = -1):
        super().__init__()
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

    def update(self):
        super().update()
        if not self.alive():
            self.id = -1
        self.lastSkeleton = np.ones_like(self.lastSkeleton) * np.nan
        return self

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
        iPart = indice(POSE_PARTS, articulationName)
        if iPart is not None:
            return self.skeleton()[iPart]
        return np.array([np.nan, np.nan])

    def articulationVisible(self, articulationName):
        iPart = indice(POSE_PARTS, articulationName)
        if iPart is not None:
            return not np.isnan(self.skeleton()[iPart]).any()
        return False

    def displaySkeleton(self, image: np.ndarray, color = (0, 255, 255)):
        skeleton = self.skeleton()
        for pair in POSE_PAIRS:
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
    # maxId = 0
    def __init__(self, pos, id = -1):
        super().__init__(25, id)
        # if id < 0:
        #     self.id = Person.maxId + 1
        # else:
        #     self.id = id
        # Person.maxId = max(Person.maxId, self.id)
        self.positions = np.array([pos])

        # self.skeleton_history = np.zeros((0, 25, 2))
        # self.lastSkeleton = np.ones((25, 2)) * np.nan
        # self.cachedSkeleton = np.ones((25, 2)) * np.nan
        # self.canUseCache = False

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
        iPart = indice(POSE_PARTS, articulationName)
        if iPart is not None:
            return self.skeleton()[iPart]
        return np.array([np.nan, np.nan])

    def articulationVisible(self, articulationName):
        iPart = indice(POSE_PARTS, articulationName)
        if iPart is not None:
            return not np.isnan(self.skeleton()[iPart]).any()
        return False

    def newPos(self, newPos):
        self.positions = np.append(self.positions, [newPos], axis=0)
        if len(self.positions) > Person.maxTrack:
            self.positions = np.delete(self.positions, 0, axis=0)

    def pos(self):
        s = self.skeleton()
        if not np.isnan(s[0]).any():
            return s[0]
        return np.mean(self.positions, axis = 0)

    def displaySkeleton(self, image, color = (0, 255, 255)):
        pos = self.pos()
        # cv2.circle(image, np.array([pos[1] * image.shape[1], pos[0] * image.shape[0]]).astype(int), 5, (0, 0, 255),
        #            -1)
        cv2.putText(image, f"{self.id}", np.array([pos[1] * image.shape[1], pos[0] * image.shape[0]]).astype(int),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2, cv2.LINE_AA)

        skeleton = self.skeleton()
        for pair in POSE_PAIRS:
            bodyA, bodyB = skeleton[pair[0]], skeleton[pair[1]]
            if not np.isnan(bodyA).any() and not np.isnan(bodyB).any():
                posA = np.array([bodyA[1] * image.shape[1], bodyA[0] * image.shape[0]]).astype(int)
                posB = np.array([bodyB[1] * image.shape[1], bodyB[0] * image.shape[0]]).astype(int)
                cv2.line(image, posA, posB, color, 2)

        for iPart in range(25):
            body = skeleton[iPart]
            if not np.isnan(body).any():
                cv2.circle(image, np.array([body[1] * image.shape[1], body[0] * image.shape[0]]).astype(int), 3,
                           (0, 255, 255), -1)

        return image



class HandDetected(SkeletonDetection):
    def __init__(self, id = -1):
        super().__init__(22, id)

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

    def displaySkeleton(self, image, color = (0, 255, 255)):
        skeleton = self.skeleton()
        for iPart in range(self.nb_articulations):
            body = skeleton[iPart]
            if not np.isnan(body).any():
                position = np.array([body[1] * image.shape[1], body[0] * image.shape[0]]).astype(int)
                cv2.circle(image, position, 3, color, -1)
                cv2.putText(image, HAND_PARTS[iPart], position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)

        for pair in HAND_PAIRS:
            bodyA, bodyB = skeleton[pair[0]], skeleton[pair[1]]
            if not np.isnan(bodyA).any() and not np.isnan(bodyB).any():
                posA = np.array([bodyA[1] * image.shape[1], bodyA[0] * image.shape[0]]).astype(int)
                posB = np.array([bodyB[1] * image.shape[1], bodyB[0] * image.shape[0]]).astype(int)
                cv2.line(image, posA, posB, color, 2)
        return image


class FaceDetected(Detection):
    maxId = 0
    def __init__(self, pos, id = -1):
        super().__init__()
        if id < 0:
            self.id = FaceDetected.maxId + 1
        else:
            self.id = id
        FaceDetected.maxId = max(FaceDetected.maxId, self.id)
        self.positions = np.array([pos])

        self.skeleton_history = np.zeros((0, 25, 2))
        self.lastSkeleton = np.ones((25, 2)) * np.nan
        self.cachedSkeleton = np.ones((25, 2)) * np.nan
        self.canUseCache = False

    def update(self):
        super().update()
        if not self.alive():
            self.positions = np.array([None, None])
            self.id = -1
        self.lastSkeleton = np.ones_like(self.lastSkeleton) * np.nan
        return self

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

    def displaySkeleton(self, image):
        skeleton = self.skeleton()
        for iPart in range(25):
            body = skeleton[iPart]
            if not np.isnan(body).any():
                cv2.circle(image, np.array([body[1] * image.shape[1], body[0] * image.shape[0]]).astype(int), 3,
                           (0, 255, 255), -1)
        return image






class ObjectDetected(Detection):
    def __init__(self, idClasse, label, box, confidence):
        super().__init__()
        self.idClasse = idClasse
        self.label = label
        self.box = box
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
        self.pose_resolution = [256, 256]
        self.max_bodies = 1
        self.pose_model_name = "body_25"
        self.pose_config_filename = "pose_deploy.prototxt"
        self.pose_weights_filename = "pose_iter_584000.caffemodel"
        self.pose_proto_file_path = self.models_paths + "/pose/" + self.pose_model_name + "/" + self.pose_config_filename
        self.pose_weights_file_path = self.models_paths + "/pose/" + self.pose_model_name + "/" + self.pose_weights_filename
        self.pose_skip_frames = 2

        self.use_body_part_detection = False
        self.cascade_front_model = "haarcascade_frontalface_default.xml"
        self.cascade_profile_model = "haarcascade_profileface.xml"
        self.cascade_smile_model = "haarcascade_smile.xml"
        self.cascade_upperbody_model = "haarcascade_fullbody.xml"
        self.cascade_fullbody_model = "haarcascade_upperbody.xml"

        self.cascade_front_path = self.models_paths + "/haarcascades/" + self.cascade_front_model
        self.cascade_profile_path = self.models_paths + "/haarcascades/" + self.cascade_profile_model
        self.cascade_smile_path = self.models_paths + "/haarcascades/" + self.cascade_smile_model
        self.cascade_fullbody_path = self.models_paths + "/haarcascades/" + self.cascade_fullbody_model
        self.cascade_upperbody_path = self.models_paths + "/haarcascades/" + self.cascade_upperbody_model
        self.body_parts_skip_frames = 2

        self.use_face = False
        self.face_model_filename = "lbfmodel.yaml"
        self.face_landmarks_path = self.models_paths + "/face_landmarks/" + self.face_model_filename
        self.face_skip_frames = 2

        self.use_hands = False
        self.hand_resolution = [256, 256]
        self.hand_config_filename = "pose_deploy.prototxt"
        self.hand_weights_filename = "pose_iter_120000.caffemodel"
        self.hand_proto_file_path = self.models_paths + "/hand/" + self.hand_config_filename
        self.hand_weights_file_path = self.models_paths + "/hand/" + self.hand_weights_filename
        self.hand_skip_frames = 3

        self.use_yolo = False
        self.yolo_resolution = [256, 256]
        self.yolo_threshold = 0.1
        self.yolo_model = "yolov4"
        self.yolo_classes_filename = "coco.names"
        self.yolo_config_path = self.models_paths + "/yolo/" + self.yolo_model + ".cfg"
        self.yolo_weights_path = self.models_paths + "/yolo/" + self.yolo_model + ".weights"
        self.yolo_classes_path = self.models_paths + "/yolo/" + self.yolo_classes_filename
        self.yolo_skip_frames = 1

        self.use_aruco = False
        self.aruco_dictionary = cv2.aruco.DICT_6X6_1000
        self.aruco_skip_frames = 0

        self.flip_image = False

        self.init_paths()

    def init_paths(self):
        self.pose_proto_file_path = self.models_paths + "/pose/" + self.pose_model_name + "/" + self.pose_config_filename
        self.pose_weights_file_path = self.models_paths + "/pose/" + self.pose_model_name + "/" + self.pose_weights_filename

        self.cascade_front_path = self.models_paths + "/haarcascades/" + self.cascade_front_model
        self.cascade_profile_path = self.models_paths + "/haarcascades/" + self.cascade_profile_model
        self.cascade_smile_path = self.models_paths + "/haarcascades/" + self.cascade_smile_model
        self.cascade_fullbody_path = self.models_paths + "/haarcascades/" + self.cascade_fullbody_model
        self.cascade_upperbody_path = self.models_paths + "/haarcascades/" + self.cascade_upperbody_model

        self.face_landmarks_path = self.models_paths + "/face_landmarks/" + self.face_model_filename

        self.hand_proto_file_path = self.models_paths + "/hand/" + self.hand_config_filename
        self.hand_weights_file_path = self.models_paths + "/hand/" + self.hand_weights_filename

        self.yolo_config_path = self.models_paths + "/yolo/" + self.yolo_model + ".cfg"
        self.yolo_weights_path = self.models_paths + "/yolo/" + self.yolo_model + ".weights"
        self.yolo_classes_path = self.models_paths + "/yolo/" + self.yolo_classes_filename

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

        if not self.use_body:
            if self.use_hands or self.use_face:
                print("Warning: Hands (use_hands) and faces (use_face) are currently "
                      "only available if use_body is set to True!")
                self.use_hands = False
                self.use_face = False

        if self.use_face:
            self.use_face = False
            print("[Warning] Unfortunately, current implementation is not efficient at all with faces. Faces are deactivated.")

        if self.use_hands:
            print("[Warning] The hand detection system is still very very bad and should not be used right now... ")

        self.face_cascade = cv2.CascadeClassifier(parameters.cascade_front_path)
        self.face_profile_cascade = cv2.CascadeClassifier(parameters.cascade_profile_path)
        self.fullbody_cascade = cv2.CascadeClassifier(parameters.cascade_fullbody_path)
        self.upperbody_cascade = cv2.CascadeClassifier(parameters.cascade_upperbody_path)
        self.smile_cascade = cv2.CascadeClassifier(parameters.cascade_smile_path)

        self.landmark_detector  = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(parameters.face_landmarks_path)

        self.pose_net = cv2.dnn.readNetFromCaffe(parameters.pose_proto_file_path, parameters.pose_weights_file_path)
        self.pose_netInputSize = np.array(parameters.pose_resolution)
        pose_netOutputSize = np.ceil(self.pose_netInputSize / 8).astype(int)
        self.pose_heatmaps = np.zeros((25, pose_netOutputSize[0], pose_netOutputSize[1]))
        self.max_bodies = parameters.max_bodies

        self.hand_net = cv2.dnn.readNetFromCaffe(parameters.hand_proto_file_path, parameters.hand_weights_file_path)
        self.hand_netInputSize = np.array(parameters.hand_resolution)
        hand_netOutputSize = np.ceil(self.hand_netInputSize / 8).astype(int)
        self.hand_heatmaps = np.zeros((22, hand_netOutputSize[0], hand_netOutputSize[1]))

        self.yolo_net = cv2.dnn.readNetFromDarknet(parameters.yolo_config_path, parameters.yolo_weights_path)
        self.yolo_netInputSize = parameters.yolo_resolution
        self.yolo_threshold = parameters.yolo_threshold

        self.yolo_classes = None
        with open(parameters.yolo_classes_path, 'rt') as f:
            self.yolo_classes = f.read().rstrip('\n').split('\n')

        if self.yolo_net.empty():
            print(f"The YOLO network has not been loaded correctly!\nPlease check if these files exists: '{parameters.yolo_config_path}' and '{parameters.yolo_weights_path}'")

        if self.pose_net.empty():
            print(f"The OpenPose network has not been loaded correctly!\nPlease check if these files exists: '{parameters.pose_protoFile_path}' and '{parameters.pose_weightsFile_path}'")

        aruco_dictionary = cv2.aruco.getPredefinedDictionary(parameters.aruco_dictionary)
        aruco_parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
        self.aruco_detector = detector

        self.persons: List[Person] = []
        self.objects_detected: List[ObjectDetected] = []
        self.arucos: List[Aruco] = []
        self.hands: List[HandDetected] = []

        self.current_frame = 0

        self.yolo_skip = parameters.yolo_skip_frames
        self.pose_skip = parameters.pose_skip_frames
        self.face_skip = parameters.face_skip_frames
        self.hand_skip = parameters.hand_skip_frames
        self.body_parts_skip = parameters.body_parts_skip_frames
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
        inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, self.yolo_netInputSize, (127.5, 127.5, 127.5), crop=False, swapRB=True)
        self.yolo_net.setInput(inpBlob)
        out = self.yolo_net.forward()
        self.objects_detected = self.detections_Yolo(img, out)
        return self.objects_detected

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

    @staticmethod
    def _call_network(image, network, input_size, normalize=True, swapRB=True):
        if normalize:
            inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, input_size, (127.5, 127.5, 127.5), swapRB=swapRB, crop=False)
        else:
            inpBlob = cv2.dnn.blobFromImage(image, 1.0, input_size, (0, 0, 0), swapRB=swapRB, crop=False)
        network.setInput(inpBlob)
        output = network.forward()
        return output[0]

    @staticmethod
    def _update_heatmaps(net_output, threshold, heatmaps):
        decay = 1.0
        big = np.array([abs(cv2.resize(out, (heatmaps[0].shape[1], heatmaps[0].shape[0]), interpolation=cv2.INTER_CUBIC)) for out in net_output])
        heatmaps = heatmaps * (1 - decay) + big * decay
        # heatmaps[heatmaps < threshold] = 0
        return heatmaps

    @staticmethod
    def _extract_peaks(heatmap, threshold, limit =  None):
        heatmap[heatmap < threshold] = 0
        peaks = (cv2.dilate(heatmap, kernel=np.ones((3, 3))) == heatmap) * (
                heatmap > threshold)
        peaks_coords = (np.argwhere(peaks))
        peaks_vals = [heatmap[x, y] for x, y in peaks_coords]
        peaks_coords = peaks_coords[np.argsort(-heatmap[tuple(peaks_coords.T)])].astype(np.float32) / np.array([heatmap.shape[0], heatmap.shape[1]])
        peaks_vals = sorted(peaks_vals, reverse=True)
        if limit is not None and limit > -1:
            peaks_vals = peaks_vals[:limit]
            peaks_coords = peaks_coords[:limit]
        return peaks_coords, peaks_vals

    def getPoses(self, img, threshold=0.1):
        image = img.copy()
        if self.flip:
            image = cv2.flip(image, 1)

        output = self._call_network(image, self.pose_net, self.pose_netInputSize)
        self.pose_heatmaps = self._update_heatmaps(output[:25], threshold, self.pose_heatmaps)
        peaks_coords, peaks_vals = self._extract_peaks(self.pose_heatmaps[0], threshold)

        self.persons = [pers.update() for pers in self.persons if pers.alive()]
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
            peaks_coords, peaks_vals = self._extract_peaks(self.pose_heatmaps[iPart], threshold)

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
        self.persons = sorted([p for p in self.persons], key = lambda p: -p.totalTimeAlive)[:self.max_bodies]
        return self.persons

    def getHands(self, img, threshold=0.0):
        image = img.copy()
        if self.flip:
            image = cv2.flip(image, 1)
        output = self._call_network(image, self.hand_net, self.hand_netInputSize, normalize=True, swapRB=False)
        self.hand_heatmaps = self._update_heatmaps(output, threshold, self.hand_heatmaps)
        max_hands = 0
        for p in self.persons:
            if p.articulationVisible("RWrist"):
                max_hands += 1
            if p.articulationVisible("LWrist"):
                max_hands += 1

        self.hands = self.hands[:max_hands]
        if len(self.hands) < max_hands:
            self.hands = self.hands + [HandDetected() for _ in range(max_hands - len(self.hands))]

        for iPart in range(self.hand_heatmaps.shape[0]):
            peaks_coords, peaks_vals = self._extract_peaks(self.hand_heatmaps[iPart], threshold, limit= 2 * max_hands)

            costMatrix = np.ones((len(self.hands), len(peaks_coords))) * np.inf
            for iHand, hand in enumerate(self.hands):
                skeleton = hand.skeleton()
                pos = skeleton[iPart]
                if np.isnan(pos).any():
                    pos = np.array([0, 0])
                otherParts = [skeleton[A] if A != iPart else skeleton[B] for A, B in HAND_PAIRS if iPart in (A, B) and not np.isnan(skeleton[A]).any() and not np.isnan(skeleton[B]).any()]
                for iCoord, coord in enumerate(peaks_coords):
                    diff = (coord - pos) ** 2
                    for other in otherParts:
                        diff += (coord - other)**2
                    sqrDist = diff[0] + diff[1]
                    costMatrix[iHand, iCoord] = sqrDist / peaks_vals[iCoord]
            try:
                row_ind, col_ind = linear_sum_assignment(costMatrix)

                for iHand, iCoord in zip(row_ind, col_ind):
                    self.hands[iHand].newSkeletonPos(iPart, peaks_coords[iCoord])
            except Exception as e:
                for hand in self.hands:
                    hand.newSkeletonPos(iPart, [None, None])
        self.hands = sorted([p for p in self.hands], key = lambda p: -p.totalTimeAlive)
        return self.hands


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
        # img = cv2.imread("/home/marc/Pictures/Webcam/2024-12-24-012445.jpg")
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