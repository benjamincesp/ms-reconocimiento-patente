import pytesseract
import numpy as np
from typing import Optional, List
from ..models.license_plate_model import ChileanLicensePlateValidator


class OCRService:
    
    OCR_CONFIGS = {
        'standard': r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        'alternative': r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        'full_page': r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    }
    
    @classmethod
    def extract_text(cls, img: np.ndarray, config_key: str = 'standard') -> str:
        try:
            config = cls.OCR_CONFIGS.get(config_key, cls.OCR_CONFIGS['standard'])
            text = pytesseract.image_to_string(img, config=config)
            return ChileanLicensePlateValidator.clean_text(text)
        except Exception as e:
            print(f"Error in text extraction: {e}")
            return ""
    
    @classmethod
    def extract_text_multiple_configs(cls, img: np.ndarray) -> Optional[str]:
        for config_name in ['standard', 'alternative']:
            text = cls.extract_text(img, config_name)
            if text and ChileanLicensePlateValidator.validate(text):
                return text
        return None
    
    @classmethod
    def extract_from_full_image(cls, img: np.ndarray) -> Optional[str]:
        try:
            full_text = cls.extract_text(img, 'full_page')
            words = full_text.split()
            
            for word in words:
                cleaned = ChileanLicensePlateValidator.clean_text(word)
                if ChileanLicensePlateValidator.validate(cleaned):
                    return cleaned
            return None
        except Exception as e:
            print(f"Error in full image OCR: {e}")
            return None
    
    @staticmethod
    def check_tesseract_availability() -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False