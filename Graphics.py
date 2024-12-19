from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Graphic:
    def __init__(self, image):
        """
        Initialize the Drawing object with an image.
        :param image: numpy array representing the image (height, width, channels)
        """
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                self.image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2BGRA)
                self.image[:,:,3] = 255
            else:
                self.image = image.copy()
        else:
            self.image = np.zeros((image[1], image[0], 4), dtype=np.uint8) # Consider that "image" is in fact a Tuple(width, height)
            self.image[:,:,3] = 0

    def _blend_with_alpha(self, position, color, alpha):
        """
        Apply alpha blending at a specific position.
        :param position: Tuple or array of coordinates (x, y).
        :param color: Tuple (B, G, R) - color to apply.
        :param alpha: Float - transparency level (0 to 1).
        """
        overlay = self.image.copy()
        overlay[position] = color
        cv2.addWeighted(src1=overlay, alpha=alpha, src2=self.image, beta=1 - alpha, gamma=0, dst=self.image)

    def get_image(self):
        return self.image

    def fill(self, color, alpha=1.0):
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        temp_image[:, :] = color
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self

    def draw_rectangle(self, start_point, end_point, color, thickness=2, alpha=1.0):
        """
        Draw a rectangle on the image with transparency.
        :param start_point: Tuple (x, y) - the starting point of the rectangle.
        :param end_point: Tuple (x, y) - the ending point of the rectangle.
        :param color: Tuple (B, G, R) - the color of the rectangle.
        :param thickness: Integer - thickness of the rectangle edges.
        :param alpha: Float - transparency of the rectangle.
        """
        # if thickness < 0 and False:
        #     # Filled rectangle
        #     top_left = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]))
        #     bottom_right = (max(start_point[0], end_point[0]), max(start_point[1], end_point[1]))
        #     for y in range(top_left[1], bottom_right[1]):
        #         for x in range(top_left[0], bottom_right[0]):
        #             self._blend_with_alpha((y, x), color, alpha)
        # else:
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        cv2.rectangle(temp_image, start_point, end_point, color, thickness)
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self

    def draw_circle(self, center, radius, color, thickness=2, alpha=1.0):
        """
        Draw a circle on the image with transparency.
        :param center: Tuple (x, y) - the center of the circle.
        :param radius: Integer - the radius of the circle.
        :param color: Tuple (B, G, R) - the color of the circle.
        :param thickness: Integer - thickness of the circle edge. Use -1 for filled circle.
        :param alpha: Float - transparency of the circle.
        """
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        cv2.circle(temp_image, center, radius, color, thickness)
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self

    def draw_text(self, text, position, font_path, font_size, color, alpha = 1.0):
        """
        Draw text on the image using a custom TTF font.
        :param text: String - the text to draw.
        :param position: Tuple (x, y) - where to put the text on the image.
        :param font_path: String - path to the .ttf font file.
        :param font_size: Integer - size of the font.
        :param color: Tuple (R, G, B) - color of the text (Pillow uses RGB).
        """
        # Convert BGR (OpenCV) image to RGB (Pillow)
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        rgb_image = cv2.cvtColor(temp_image, cv2.COLOR_BGRA2RGBA)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=color)

        temp_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self


    def draw_arrow(self, start_point, end_point, color, thickness=2, tip_length=0.1, alpha = 1.0):
        """
        Draw an arrow on the image.
        :param start_point: Tuple (x, y) - the starting point of the arrow.
        :param end_point: Tuple (x, y) - the ending point of the arrow.
        :param color: Tuple (B, G, R) - the color of the arrow.
        :param thickness: Integer - thickness of the arrow line.
        :param tip_length: Float - relative length of the arrow tip.
        """
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        cv2.arrowedLine(temp_image, start_point, end_point, color, thickness, tipLength=tip_length)
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self


    def resize(self, new_size, interpolation=cv2.INTER_LINEAR):
        """
        Resize the image to a new size.
        :param new_size: Tuple (new_width, new_height)
        :param interpolation: Interpolation method
        """
        self.image = cv2.resize(self.image, new_size, interpolation=interpolation)
        return self

    def crop(self, start_point, end_point):
        """
        Crop the image to the specified rectangle.
        :param start_point: Tuple (x1, y1) - the top-left coordinate of the rectangle
        :param end_point: Tuple (x2, y2) - the bottom-right coordinate of the rectangle
        """
        self.image = self.image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        return self

    def rotate(self, angle, center=None, scale=1.0):
        """
        Rotate the image around a center point at a given angle.
        :param angle: float - angle of rotation in degrees
        :param center: Tuple (x, y) - the center of rotation, defaults to the center of the image
        :param scale: float - scale factor
        """
        (h, w) = self.image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.image = cv2.warpAffine(self.image, M, (w, h))
        return self

    def apply_perspective_transform(self, src_points, dst_points, size):
        """
        Apply a perspective transformation to the image.
        :param src_points: List of four source points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        :param dst_points: List of four destination points after transformation
        :param size: Tuple (width, height) of the resulting image
        """
        matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype='float32'),
                                             np.array(dst_points, dtype='float32'))
        self.image = cv2.warpPerspective(self.image, matrix, size)
        return self

    def adjust_brightness(self, value):
        """
        Adjust the brightness of the image.
        :param value: Integer, increase or decrease brightness by this value
        """
        self.image = cv2.add(self.image, np.array([value, value, value]))
        return self

    def adjust_contrast(self, factor):
        """
        Adjust the contrast of the image.
        :param factor: Float, factor by which to multiply the contrast
        """
        self.image = cv2.multiply(self.image, np.array([factor, factor, factor]))
        return self

    def adjust_saturation(self, factor):
        """
        Adjust the saturation of the image.
        :param factor: Float, factor by which to multiply the saturation
        """
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = cv2.multiply(hsv_image[..., 1], np.array([factor]))
        self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return self

    def convert_color_space(self, conversion_code):
        """
        Convert the image to a different color space using a conversion code.
        :param conversion_code: OpenCV color conversion code (e.g., cv2.COLOR_BGR2GRAY)
        """
        self.image = cv2.cvtColor(self.image, conversion_code)
        return self


    def apply_blur(self, kernel_size=(5, 5)):
        """
        Apply a simple Gaussian blur to the image.
        :param kernel_size: Tuple indicating the size of the Gaussian kernel
        """
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)
        return self

    def apply_sharpen(self):
        """
        Apply a sharpening filter to the image.
        """
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        self.image = cv2.filter2D(self.image, -1, kernel)
        return self

    def apply_edge_detection(self):
        """
        Detect edges in the image using the Canny edge detector.
        """
        self.image = cv2.Canny(self.image, 100, 200)
        return self

    def apply_custom_filter(self, kernel):
        """
        Apply a custom kernel for various effects.
        :param kernel: A numpy array representing the filter kernel
        """
        self.image = cv2.filter2D(self.image, -1, kernel)
        return self

    def apply_sketch_effect(self):
        """
        Convert the image to a pencil sketch-like effect.
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        inverted_img = cv2.bitwise_not(gray_image)
        blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
        inverted_blur = cv2.bitwise_not(blurred_img)
        self.image = cv2.cvtColor(cv2.divide(gray_image, inverted_blur, scale=256.0), cv2.COLOR_GRAY2BGRA)
        return self

    def apply_cartoon_effect(self):
        """
        Apply a cartoon effect to the image by enhancing edges and reducing the color palette.
        """
        # Step 1: Apply a bilateral filter to reduce the color palette
        alpha = self.image[:,:,3]
        color = cv2.bilateralFilter(cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR), d=9, sigmaColor=75, sigmaSpace=75)
        # Step 2: Convert to grayscale and apply median blur
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        # Step 3: Create an edge mask using adaptive thresholding
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=9, C=2)
        # Step 4: Combine edges with the color image
        self.image = cv2.cvtColor(cv2.bitwise_and(color, color, mask=edges), cv2.COLOR_BGR2BGRA)
        self.image[:,:,3] = alpha
        return self

    def apply_painting_effect(self):
        """
        Apply a painting effect using a combination of filters.
        """
        # Step 1: Use a bilateral filter to simulate a watercolor effect
        alpha = self.image[:,:,3]
        self.image = cv2.cvtColor(cv2.stylization(cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR), sigma_s=60, sigma_r=0.6), cv2.COLOR_BGR2BGRA)
        self.image[:,:,3] = alpha
        return self



class SceneRender:
    def __init__(self, base_size):
        """
        Initialize the LayerManager with a specified base canvas size.
        :param base_size: Tuple (height, width), the size of the base canvas.
        """
        self.base_size = base_size
        self.layers: List[Tuple[Graphic, Tuple[int, int], float]] = []  # Stores layers as tuples of (image, position, alpha)

    def add_layer(self, image, position=(0, 0), alpha = 1.0):
        """
        Add a new layer with a specified position on the base canvas.
        :param image: numpy array representing the image (layer) to be added.
        :param position: Tuple (x, y), the top-left corner where the layer will be placed.
        """
        self.layers.append((image if isinstance(image, Graphic) else Graphic(image), position, alpha))

    def get_image(self):
        """
        Merge all layers onto a base canvas and return the final image.
        :return: numpy array, the merged final image.
        """

        # Create a blank canvas
        final_image = np.zeros((self.base_size[1], self.base_size[0], 3), dtype=np.uint8)

        for imageManager, position, alphaCoef in self.layers:
            image = imageManager.get_image()
            # Determine the region of the canvas that the layer will occupy
            y_start = position[1]
            y_end = y_start + image.shape[0]
            x_start = position[0]
            x_end = x_start + image.shape[1]

            # Check boundaries and adjust if necessary
            if x_end > self.base_size[0]:
                x_end = self.base_size[0]
                image = image[:, :x_end - x_start]
            if y_end > self.base_size[1]:
                y_end = self.base_size[1]
                image = image[:y_end - y_start, :]

            # Place the image on the final canvas
            alpha = alphaCoef * np.reshape(image[:,:,3].astype(float), (image.shape[0], image.shape[1], 1)) / 255.0
            final_image[y_start:y_end, x_start:x_end] = (image[:,:,:3] * (alpha) + final_image[y_start:y_end, x_start:x_end] * (1 - alpha)).astype(np.uint8)

        return final_image

    def clear(self):
        self.layers.clear()





def main():
    # Create a blank white image
    height, width = 400, 400

    cap = cv2.VideoCapture(0)

    smiley = cv2.imread("smiley.png", cv2.IMREAD_UNCHANGED)

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.resize(image, (width, height))

        layers = SceneRender((300, 300))

        # Initialize the Drawing class
        drawing = Graphic((400, 400))
        # Initialize the Drawing class
        transforms = Graphic(image.copy())

        sprite = Graphic(smiley)

        # Draw shapes and text
        drawing.draw_rectangle((0, 0), (400, 400), (255, 255, 255), -1)
        drawing.draw_rectangle((50, 50), (150, 150), (255, 0, 0), 5, alpha = 0.5)
        drawing.draw_circle((200, 200), 50, (0, 255, 255), -1, alpha=0.5)
        drawing.draw_circle((250, 200), 50, (0, 255, 0), -1, alpha=0.5)
        drawing.draw_circle((300, 200), 50, (0, 255, 0), -1, alpha=0.5)
        drawing.draw_text('Hello, OpenCV!', (50, 250), 'Fonts/Hollster.ttf', 30, (0, 0, 0), alpha = 0.8)
        drawing.draw_arrow((100, 200), (300, 300), (0, 0, 255), 3, 0.3)

        sprite.resize((300, 300), cv2.INTER_NEAREST)


        transforms.apply_perspective_transform(
            [(0, 0), (width, 0), (width, height), (0, height)],  # source points
            [(0, height // 3), (width, 0), (width, height), (0, 2 * height // 3)],  # destination points
            (width, height)
        ).apply_sketch_effect()

        # layers.add_layer(transforms, (0, 0), 0.5)
        layers.add_layer(drawing, (0, 0))
        # layers.add_layer(sprite, (0, 0))

        # Display the image
        cv2.imshow('Drawing', layers.get_image())

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
