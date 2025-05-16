import cv2
import os
import numpy as np
from datetime import datetime
from source.utils import Utils
from skimage import morphology
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
 


class ImagePreProcessor:
    def __init__(self, output_dir="preprocessed"):
        """Initialzies ImagePreProcessor with necessary components"""
        self.utils = Utils()
        self.output_dir = output_dir
        self.base_dir = self.utils.get_file_path_of_parent()
        self.output_path = os.path.join(self.base_dir, self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

    
    def load_image(self, image_path: str):
        """Safely loads an image from disk"""
        if not image_path:
            print("Invalid image path.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image from {image_path}")
        return image

    def save_image(self, image, output_path: str, label: str = ""):
        """Saves an image to disk with a timestamp to avoid overwriting"""
        # 🕒 Generate timestamp and append to filename
        base, ext = os.path.splitext(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}_{timestamp}{ext}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, image)

        if success:
            print(f"{label} image saved to {output_path}")
        else:
            print(f"Failed to save {label} image to {output_path}")

        return output_path

    def clean_mask(self, mask: np.ndarray, min_size=100, radius=3) -> np.ndarray:
        """Applies morphological cleaning: small object removal, closing, hole filling."""
        binary = mask > 0
        cleaned = morphology.remove_small_objects(binary, min_size=min_size)
        closed = morphology.binary_closing(cleaned, disk(radius))
        filled = binary_fill_holes(closed)
        return (filled * 255).astype(np.uint8)

      # --- Core Processing Methods ---
    def blurring(self, image, ksize=(5, 5), sigmaX=1.0):
        """Applies Gaussian blur to reduce noise."""
        output = image.copy()
        return cv2.GaussianBlur(output, ksize, sigmaX)

    def canny_edge_detecting(self, image, thresh1=50, thresh2=150):
        """Applies Canny edge detection with configurable thresholds."""
        output = image.copy()
        return cv2.Canny(output, thresh1, thresh2)
    
    def thresholding(self, image, lowerb=(0, 0, 50), upperb=(180, 255, 255), 
                 lowerg=(40, 50, 50), upperg=(80, 255, 255), 
                 lowerred1=(0, 100, 100), upperred1=(10, 255, 255), 
                 lowerred2=(160, 100, 100), upperred2=(180, 255, 255)):
        """
        Applies HSV inRange thresholding. Returns cleaned binary masks for
        blue, green, and red objects in the image.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red thresholds (red is split into two ranges due to hue wrapping)
        red1 = cv2.inRange(hsv, lowerred1, upperred1)
        red2 = cv2.inRange(hsv, lowerred2, upperred2)
        red_mask = cv2.bitwise_or(red1, red2)

         # Green thresholds
        green_mask = cv2.inRange(hsv, lowerg, upperg)

        # Blue thresholds
        blue_mask = cv2.inRange(hsv, lowerb, upperb)

        return (
            self.clean_mask(blue_mask, min_size=100),
            self.clean_mask(green_mask, min_size=200),
            self.clean_mask(red_mask, min_size=100),
        )


    def segmentation(self, original_img, blue_mask, green_mask, red_mask,
                 mode=cv2.RETR_EXTERNAL, contour_method=cv2.CHAIN_APPROX_SIMPLE,
                 contourIdx=-1, thickness=2):
        image = original_img.copy()
        print(f"Mode: {mode}\n Contour METHOD: {contour_method}\n contourIdx:{contourIdx}\n thickness: {thickness}\n")

        if np.any(blue_mask):
            nuclei_contours, _ = cv2.findContours(blue_mask, mode, contour_method)
            cv2.drawContours(image, nuclei_contours, contourIdx, (0, 0, 255), thickness)

        if np.any(green_mask):
            cell_contours, _ = cv2.findContours(green_mask, mode, contour_method)
            cv2.drawContours(image, cell_contours, contourIdx, (255, 0, 255), thickness)

        if np.any(red_mask):
            red_contours, _ = cv2.findContours(red_mask, mode, contour_method)
            cv2.drawContours(image, red_contours, contourIdx, (255, 255, 0), thickness)

        return image

    def contouring(self, image, edges_img, mode=cv2.RETR_EXTERNAL, contour_method=cv2.CHAIN_APPROX_SIMPLE,
               contourIdx=-1, color=(0, 255, 0), thickness=2):
        """
        Draws contours from a binary edge image onto a color image.

        Args:
            image (np.ndarray): Original BGR image.
            edges_img (np.ndarray): Binary edge map (from Canny or thresholding).
            mode (int): Contour retrieval mode.
            contour_method (int): Contour approximation method.
            contourIdx (int): Index of contour to draw (-1 = all).
            color (tuple): BGR color for drawing contours.
            thickness (int): Line thickness.

        Returns:
            np.ndarray: Image with contours drawn.
        """
        overlay = image.copy()
        contours, _ = cv2.findContours(edges_img, mode, contour_method)
        cv2.drawContours(overlay, contours, contourIdx, color, thickness)
        return overlay


    # def apply_blurring_only(self, image_path:str):
    #     """Loads an image, applies Gaussian blur, and saves output"""
    #     image = self.load_image(image_path)
    #     if image is None:
    #         return None

    #     blurred = self.blurring(image)

    #     filename = f"blurred_{os.path.basename(image_path)}"
    #     output_path = os.path.join(self.output_path, filename)
    #     return self.save_image(blurred, output_path, label="Blurred")


    # def apply_thresholding_only(self, image_path: str, thresh=70, maxVal=255, threshold_type=cv2.THRESH_BINARY):
    #     """Applies grayscale + thresholding to an image"""
    #     image = self.load_image(image_path)
    #     if image is None:
    #         return None


    #     blue_m, green_m, red_m = self.thresholding(image)

    #     red_colored = cv2.merge([np.zeros_like(red_m), np.zeros_like(red_m), red_m])
    #     green_colored = cv2.merge([np.zeros_like(green_m), green_m, np.zeros_like(green_m)])
    #     blue_colored = cv2.merge([blue_m, np.zeros_like(blue_m), np.zeros_like(blue_m)])

    #     combined_mask = cv2.addWeighted(red_colored, 1.0, green_colored, 1.0, 0)
    #     combined_mask = cv2.addWeighted(combined_mask, 1.0, blue_colored, 1.0, 0)

    #     overlay = cv2.addWeighted(image, 0.6, combined_mask, 0.4, 0)

    #     filename = f"thresholded_{os.path.basename(image_path)}"
    #     output_path = os.path.join(self.output_path, filename)
    #     return self.save_image(overlay, output_path, label="Thresholded")
    

    # def apply_canny_only(self, image_path: str, thresh1=50, thresh2=150):
    #     """Applies grayscale + canny edge detection to an image"""
    #     image = self.load_image(image_path)
    #     if image is None:
    #         return None

    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.Canny(gray, thresh1, thresh2)

    #     overlay = image.copy()
    #     overlay[edges != 0] = [0, 255, 0] 

    #     filename = f"Canny_{os.path.basename(image_path)}"
    #     output_path = os.path.join(self.output_path, filename)
    #     return self.save_image(overlay, output_path, label="Canny")

    
    def apply_segmentation_only(self, image_path: str):
        """Finds and outlines contours in an image after thresholding"""
        image = self.load_image(image_path)
        if image is None:
            return None

        segmented = self.segmentation(image)

        filename = f"Segmented_{os.path.basename(image_path)}"
        output_path = os.path.join(self.output_path, filename)
        return self.save_image(segmented, output_path, label="Segmented")
    
    def is_valid_combination(self, param_dict: dict) -> bool:
        """
        Validates a parameter combination based on known constraints.
        Assumes a flat dictionary of parameters passed to apply_default_preprocessing().
        """

        # ✅ 1. Canny: threshold1 < threshold2
        if "threshold1" in param_dict and "threshold2" in param_dict:
            if param_dict["threshold1"] >= param_dict["threshold2"]:
                return False

        # ✅ 2. addWeighted: alpha + beta ≈ 1.0
        if "alpha" in param_dict and "beta" in param_dict:
            total = param_dict["alpha"] + param_dict["beta"]
            if not (0.95 <= total <= 1.05):  # Allow float tolerance
                return False

        # ✅ 3. Gaussian Blur: ksize must be a tuple of odd positive ints
        if "ksize" in param_dict:
            k = param_dict["ksize"]
            if not (isinstance(k, tuple) and len(k) == 2 and all(isinstance(v, int) and v % 2 == 1 and v > 1 for v in k)):
                return False

        # ✅ 4. HSV bounds: lowerb must be <= upperb (component-wise)
        if "lowerb" in param_dict and "upperb" in param_dict:
            lb = param_dict["lowerb"]
            ub = param_dict["upperb"]
            if not all(l <= u for l, u in zip(lb, ub)):
                return False

        return True


    def apply_default_preprocessing(self, image_path:str, **kwargs):
        """Applies threshold, blur, canny, segmentation, and saves image"""
        image = self.load_image(image_path)
        if image is None:
            return None


        grouped_params = kwargs

        # --- Step 1: Thresholding ---
        blue_mask, green_mask, red_mask = self.thresholding(
            image,
            lowerb=grouped_params["in_range"].get("lowerb", (0, 0, 50)),
            upperb=grouped_params["in_range"].get("upperb", (180, 255, 255)),
            lowerg=grouped_params["in_range"].get("lowerg", (40, 50, 50)),
            upperg=grouped_params["in_range"].get("upperg", (80, 255, 255)),
            lowerred1=grouped_params["in_range"].get("lowerred1", (0, 100, 100)),
            upperred1=grouped_params["in_range"].get("upperred1", (10, 255, 255)),
            lowerred2=grouped_params["in_range"].get("lowerred2", (160, 100, 100)),
            upperred2=grouped_params["in_range"].get("upperred2", (180, 255, 255))
        )

        # --- Step 2: Blur the masks ---
        ksize = grouped_params["gaussian_blur"].get("ksize", (5, 5))
        sigmaX = grouped_params["gaussian_blur"].get("sigmaX", 1.0)

        # blurred = self.blurring(image)


        # blue_mask = self.blurring(blue_mask, ksize=ksize, sigmaX=sigmaX)
        # green_mask = self.blurring(green_mask, ksize=ksize, sigmaX=sigmaX)
        # red_mask = self.blurring(red_mask, ksize=ksize, sigmaX=sigmaX)

        #--- Step 3: Canny edge detection on blurred image ---
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        blurred_image = self.blurring(gray, ksize=ksize, sigmaX=sigmaX)
        edges = self.canny_edge_detecting(
            blurred_image,
            thresh1=grouped_params["canny"].get("threshold1", 50),
            thresh2=grouped_params["canny"].get("threshold2", 150)
        )

        # --- Step 4: Segmentation from masks ---
        segmented = self.segmentation(
            image,
            blue_mask=blue_mask,
            green_mask=green_mask,
            red_mask=red_mask,
            mode=grouped_params["find_contours"].get("mode", cv2.RETR_EXTERNAL),
            contour_method=grouped_params["find_contours"].get("contour_method", cv2.CHAIN_APPROX_SIMPLE),
            contourIdx=grouped_params["draw_contours"].get("contourIdx", -1),
            thickness=grouped_params["draw_contours"].get("thickness", 2),
        )

        #--- Step 5: Contour overlay from Canny edges ---
        contoured = self.contouring(
            segmented,
            edges_img=edges,
            mode=grouped_params["find_contours"].get("mode", cv2.RETR_EXTERNAL),
            contour_method=grouped_params["find_contours"].get("contour_method", cv2.CHAIN_APPROX_SIMPLE),
            contourIdx=grouped_params["draw_contours"].get("contourIdx", -1),
            color=grouped_params["draw_contours"].get("color", (0, 255, 0)),
            thickness=grouped_params["draw_contours"].get("thickness", 2),
        )

        # --- Step 6: Blend with original image ---
        alpha = grouped_params["add_weighted"].get("alpha", 0.7)
        beta = grouped_params["add_weighted"].get("beta", 0.3)
        gamma = grouped_params["add_weighted"].get("gamma", 0.0)
        overlay = cv2.addWeighted(image, alpha, segmented, beta, gamma)

        filename = f"Default_Preprocessed{os.path.basename(image_path)}"
        output_path = os.path.join(self.output_path, filename)
        return self.save_image(overlay, output_path, label="Default Preprocessing")
    
    def preprocess_image(self, image_path, method, **kwargs):
        """Determine which preprocessing function to call and returns the result of that function"""
        match method:
            case "default":
                return self.apply_default_preprocessing(image_path, **kwargs)
            # case "blur":
            #     return self.apply_blurring_only(image_path)
            # case "threshold":
            #     return self.apply_thresholding_only(image_path, **kwargs)
            # case "canny":
            #     return self.apply_canny_only(image_path, **kwargs)
            # case "segmented":
            #     return self.apply_segmentation_only(image_path)
            case "none":
                return image_path
            case _:
                return "Unknown preprocessing method."
 
