from flask import jsonify
from ..services.ocr_service import OCRService


class HealthController:
    
    @staticmethod
    def health_check():
        tesseract_available = OCRService.check_tesseract_availability()
        
        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "service": "license-plate-recognition",
            "algorithm": "santifiorino-lpr",
            "tesseract_available": tesseract_available
        }), 200