import cv2
import os
import numpy as np
from source.utils import Utils
 


class ImagePreProcessor:
    def __init__(self, output_dir="preprocessed"):
        """Initialzies ImagePreProcessor with necessary components"""
        self.utils = Utils()
        self.output_dir = output_dir
        self.base_dir = self.utils.get_file_path_of_parent()
        self.output_path = os.path.join(self.base_dir, self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

      # --- Core Processing Methods ---
    def blurring(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def canny_edge_detecting(self, image):
        edges = cv2.Canny(image, 50, 150)
        return edges


    def thresholding(self, image):
        _, thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def segmentation(self, original_img, binary_img):
        output_img = original_img.copy()
        contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 5:
                cv2.circle(output_img, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        return output_img

    def apply_blurring_only(self, image_path:str):
        """Loads an image, applies Gaussian blur, and saves output"""
        if not image_path:
            print("Invalid image path.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image.")
            return None

        blurred = self.blurring(image)

        path_to_output = os.path.join(self.output_path, os.path.basename(image_path))
        cv2.imwrite(path_to_output, blurred)
        print(f"Blurred image saved to {path_to_output}")
        return path_to_output


    def apply_thresholding_only(self, image_path:str):
        """Applies grayscale + thresholding to an image"""
        if not image_path:
            print("Invalid image path.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image.")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = self.thresholding(gray)

        path_to_output = os.path.join(self.output_path, os.path.basename(image_path))
        cv2.imwrite(path_to_output, thresh)
        print(f"Thresholded image saved to {path_to_output}")
        return path_to_output
    

    def apply_canny_only(self, image_path:str):
        """Applies grayscale + canny edge detection to an image"""
        if not image_path:
            print("Invalid image path.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image.")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = self.canny_edge_detecting(gray)

        path_to_output = os.path.join(self.output_path, os.path.basename(image_path))
        cv2.imwrite(path_to_output, thresh)
        print(f"Canny image saved to {path_to_output}")
        return path_to_output

    
    def apply_segmentation_only(self, image_path:str):
        """Finds and outlines contours in an image after thresholding"""
        if not image_path:
            print("Invalid image path.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image.")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = self.thresholding(gray)
        segmented = self.segmentation(image, thresh)

        path_to_output = os.path.join(self.output_path, os.path.basename(image_path))
        cv2.imwrite(path_to_output, segmented)
        print(f"Segmented image saved to {path_to_output}")
        return path_to_output


    def apply_default_preprocessing(self, image_path:str):
        """Applies grayscale, blur, threshold, segmentation, and saves image"""
        if not image_path:
            print("Invalid image path.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image.")
            return None

        blurred = self.blurring(image)
        canny = self.canny_edge_detecting(blurred)
        #thresh = self.thresholding(blurred)
        segmented = self.segmentation(image, canny)

        path_to_output = os.path.join(self.output_path, os.path.basename(image_path))
        cv2.imwrite(path_to_output, segmented)
        print(f"Fully preprocessed image saved to {path_to_output}")
        return path_to_output
 
