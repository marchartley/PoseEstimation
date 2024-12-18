import cv2
import time

class FPSCounter:
    def __init__(self):
        self.startTime = time.time()
        self.frames = 0
        self.fps = 0

    def update(self):
        self.frames += 1
        if self.frames > 20:
            self.fps = self.frames / (time.time() - self.startTime)
            self.startTime = time.time()
            self.frames = 0

    def display(self, img, position = (10, 30)):
        cv2.putText(img, f"FPS: {round(self.fps)}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img