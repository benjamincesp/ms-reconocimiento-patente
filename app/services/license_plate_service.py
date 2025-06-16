import cv2
from typing import Optional
from ..models.license_plate_model import LicensePlateResult, ImageProcessingParams, ChileanLicensePlateValidator
from .image_processing_service import ImageProcessingService
from .ocr_service import OCRService


class LicensePlateService:
    
    def __init__(self):
        self.processing_params = ImageProcessingParams.default()
        self.image_processor = ImageProcessingService()
        self.ocr_service = OCRService()
    
    def recognize_santifiorino_method(self, img_bytes: bytes) -> LicensePlateResult:
        try:
            img = self.image_processor.decode_image(img_bytes)
            if img is None:
                return LicensePlateResult(None, 0.0, "santifiorino", "Failed to decode image")
            
            print(f"Processing image of size: {img.shape}")
            gray_img = self.image_processor.grayscale(img)
            
            for threshold in self.processing_params.threshold_values:
                print(f"Trying threshold: {threshold}")
                binary_img = self.image_processor.apply_threshold(gray_img, threshold)
                contours = self.image_processor.find_contours(binary_img)
                
                license_plates = []
                for contour in contours:
                    if self.image_processor.is_license_plate(
                        contour, 
                        self.processing_params.min_aspect_ratio,
                        self.processing_params.max_aspect_ratio,
                        self.processing_params.min_width,
                        self.processing_params.min_height
                    ):
                        license_plates.append(contour)
                
                for i, plate_contour in enumerate(license_plates):
                    license_plate_img = self.image_processor.crop_license_plate(gray_img, plate_contour)
                    
                    if license_plate_img.size == 0:
                        continue
                    
                    processed_img = self.image_processor.process_license_plate(license_plate_img)
                    extracted_text = self.ocr_service.extract_text(processed_img)
                    
                    if ChileanLicensePlateValidator.validate(extracted_text):
                        return LicensePlateResult(extracted_text, 0.9, "santifiorino", None)
                    
                    inverted_only = cv2.bitwise_not(license_plate_img)
                    text2 = self.ocr_service.extract_text(inverted_only)
                    
                    if ChileanLicensePlateValidator.validate(text2):
                        return LicensePlateResult(text2, 0.8, "santifiorino_alt", None)
                    
                    text3 = self.ocr_service.extract_text(license_plate_img)
                    
                    if ChileanLicensePlateValidator.validate(text3):
                        return LicensePlateResult(text3, 0.7, "santifiorino_orig", None)
            
            return LicensePlateResult(None, 0.0, "santifiorino", "No valid license plate found")
            
        except Exception as e:
            return LicensePlateResult(None, 0.0, "santifiorino", f"Error in recognition: {e}")
    
    def recognize_contour_method(self, img_bytes: bytes) -> LicensePlateResult:
        try:
            img = self.image_processor.decode_image(img_bytes)
            if img is None:
                return LicensePlateResult(None, 0.0, "contour", "Failed to decode image")
            
            img = self.image_processor.resize_if_large(img)
            gray, edges = self.image_processor.bilateral_filter_preprocessing(img)
            plate_contour = self.image_processor.find_rectangular_contours(edges, img)
            
            if plate_contour is not None:
                cropped = self.image_processor.extract_contour_region(gray, plate_contour)
                if cropped is not None:
                    plate_text = self.ocr_service.extract_text_multiple_configs(cropped)
                    if plate_text:
                        return LicensePlateResult(plate_text, 0.85, "contour", None)
            
            enhanced = self.image_processor.enhance_image(gray)
            full_text_result = self.ocr_service.extract_from_full_image(enhanced)
            
            if full_text_result:
                return LicensePlateResult(full_text_result, 0.6, "contour_full", None)
            
            return LicensePlateResult(None, 0.0, "contour", "No valid license plate found")
            
        except Exception as e:
            return LicensePlateResult(None, 0.0, "contour", f"Error in recognition: {e}")
    
    def recognize(self, img_bytes: bytes) -> LicensePlateResult:
        santifiorino_result = self.recognize_santifiorino_method(img_bytes)
        
        if santifiorino_result.plate and santifiorino_result.confidence > 0.7:
            return santifiorino_result
        
        contour_result = self.recognize_contour_method(img_bytes)
        
        if contour_result.plate and contour_result.confidence > santifiorino_result.confidence:
            return contour_result
        
        return santifiorino_result if santifiorino_result.plate else contour_result