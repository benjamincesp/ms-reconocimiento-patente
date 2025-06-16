from dataclasses import dataclass
from typing import Optional, List, Tuple
import re


@dataclass
class LicensePlateResult:
    plate: Optional[str]
    confidence: float = 0.0
    processing_method: str = ""
    error: Optional[str] = None


@dataclass
class ImageProcessingParams:
    threshold_values: List[int]
    min_aspect_ratio: float
    max_aspect_ratio: float
    min_width: int
    min_height: int
    
    @classmethod
    def default(cls):
        return cls(
            threshold_values=[170, 150, 190, 130, 210],
            min_aspect_ratio=2.0,
            max_aspect_ratio=6.0,
            min_width=100,
            min_height=30
        )


class ChileanLicensePlateValidator:
    PATTERNS = [
        r'^[A-Z]{4}[0-9]{2}$',  # AAAA##
        r'^[A-Z]{2}[0-9]{4}$',  # AA####
        r'^[A-Z]{3}[0-9]{3}$',  # AAA###
    ]
    
    @classmethod
    def validate(cls, text: str) -> bool:
        if not text or len(text) < 5:
            return False
        
        for pattern in cls.PATTERNS:
            if re.match(pattern, text):
                return True
        
        letter_count = sum(1 for c in text if c.isalpha())
        number_count = sum(1 for c in text if c.isdigit())
        
        return letter_count >= 2 and number_count >= 1
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        return re.sub(r'[^A-Z0-9]', '', text.upper().strip())