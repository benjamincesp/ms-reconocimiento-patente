import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.segmentation import clear_border


class ImageProcessingService:
    
    @staticmethod
    def decode_image(img_bytes: bytes) -> Optional[np.ndarray]:
        try:
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    @staticmethod
    def grayscale(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def apply_threshold(img: np.ndarray, threshold_value: int = 170) -> np.ndarray:
        return cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
    
    @staticmethod
    def find_contours(img: np.ndarray) -> List:
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def is_license_plate(contour, min_aspect_ratio: float, max_aspect_ratio: float, 
                        min_width: int, min_height: int) -> bool:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        return (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and 
                w >= min_width and h >= min_height)
    
    @staticmethod
    def crop_license_plate(img: np.ndarray, contour) -> np.ndarray:
        x, y, w, h = cv2.boundingRect(contour)
        return img[y:y+h, x:x+w]
    
    @staticmethod
    def process_license_plate(license_plate_img: np.ndarray) -> np.ndarray:
        cleared = clear_border(license_plate_img)
        inverted = cv2.bitwise_not(cleared)
        return inverted
    
    @staticmethod
    def bilateral_filter_preprocessing(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(bilateral, 30, 200)
        return gray, edges
    
    @staticmethod
    def find_rectangular_contours(edges: np.ndarray, original_image: np.ndarray):
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            
            if len(approx) == 4:
                return approx
        return None
    
    @staticmethod
    def extract_contour_region(image: np.ndarray, contour) -> Optional[np.ndarray]:
        if contour is None:
            return None
        
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        (x, y) = np.where(mask == 255)
        if len(x) == 0 or len(y) == 0:
            return None
            
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        
        cropped = image[topx:bottomx+1, topy:bottomy+1]
        
        if cropped.size == 0:
            return None
        
        if cropped.shape[0] < 50 or cropped.shape[1] < 150:
            scale_factor = max(50 / cropped.shape[0], 150 / cropped.shape[1])
            new_width = int(cropped.shape[1] * scale_factor)
            new_height = int(cropped.shape[0] * scale_factor)
            cropped = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cropped
    
    @staticmethod
    def enhance_image(img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)
    
    @staticmethod
    def resize_if_large(image: np.ndarray, max_width: int = 800) -> np.ndarray:
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image