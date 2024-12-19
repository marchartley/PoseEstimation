import cv2
import time

class FPSCounter:
    def __init__(self):
        self.FPS_limit = None
        self.startTime = time.time()
        self.frames = 0
        self.fps = 0
        self.lastTime = time.time()
        self.dt = 0

    def update(self):
        currentTime = time.time()
        self.dt = currentTime - self.lastTime
        # if self.FPS_limit is not None and self.FPS_limit > 0 and self.fps > self.FPS_limit:
        #     time.sleep(0.1)
        self.lastTime = time.time()
        self.frames += 1
        if self.frames > 10:
            self.fps = self.frames / (currentTime - self.startTime)
            self.startTime = currentTime
            self.frames = 0

    def display(self, img, position = (10, 30)):
        cv2.putText(img, f"FPS: {round(self.fps)}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

    def limitFPS(self, maxFPS = None):
        self.FPS_limit = maxFPS