from flask import request, jsonify
from ..services.license_plate_service import LicensePlateService
from ..services.gcp_vision_service import GCPVisionService
from ..services.database_service import DatabaseService
from ..utils.validators import FileValidator


class LicensePlateController:
    
    def __init__(self):
        self.license_plate_service = LicensePlateService()
        self.gcp_vision_service = GCPVisionService()
        self.database_service = DatabaseService()
        self.file_validator = FileValidator()
    
    def detect_license_plate(self):
        try:
            validation_result = self.file_validator.validate_image_upload(request)
            if not validation_result['valid']:
                return jsonify({
                    "placa": None, 
                    "isOnDatabase": 0,
                    "status": validation_result['status_code'],
                    "error": validation_result['error'],
                    "method": "opencv_tesseract"
                }), 200
            
            file = request.files['image']
            image_bytes = file.read()
            
            print(f"\nProcessing image: {file.filename}")
            
            result = self.license_plate_service.recognize(image_bytes)
            
            if result.plate:
                print(f"✅ Detection successful: {result.plate} (method: {result.processing_method})")
                
                # Check if license plate exists in database
                db_result = self.database_service.check_license_plate_exists(result.plate)
                
                is_on_database = 1 if db_result['exists'] else 0
                status_code = 200 if db_result['exists'] else 404
                
                return jsonify({
                    "placa": result.plate,
                    "isOnDatabase": is_on_database,
                    "status": status_code,
                    "method": "opencv_tesseract"
                }), 200
            else:
                print(f"❌ No license plate detected (method: {result.processing_method})")
                if result.error:
                    print(f"Error: {result.error}")
                return jsonify({
                    "placa": None,
                    "isOnDatabase": 0,
                    "status": 404,
                    "method": "opencv_tesseract"
                }), 200
            
        except Exception as e:
            print(f"❌ Controller error: {e}")
            return jsonify({
                "placa": None, 
                "isOnDatabase": 0,
                "status": 500,
                "error": "Internal server error",
                "method": "opencv_tesseract"
            }), 200
    
    def detect_license_plate_v2(self):
        """
        Version 2: Uses Google Cloud Vision API for license plate detection
        """
        try:
            validation_result = self.file_validator.validate_image_upload(request)
            if not validation_result['valid']:
                return jsonify({
                    "placa": None, 
                    "error": validation_result['error'],
                    "method": "gcp_vision"
                }), validation_result['status_code']
            
            file = request.files['image']
            image_bytes = file.read()
            
            print(f"\nProcessing image with GCP Vision: {file.filename}")
            
            # Use GCP Vision API to extract license plate
            license_plate = self.gcp_vision_service.extract_license_plate_from_image(image_bytes)
            
            if license_plate:
                print(f"✅ GCP Vision detection successful: {license_plate}")
                
                # Check if license plate exists in database
                db_result = self.database_service.check_license_plate_exists(license_plate)
                
                is_on_database = 1 if db_result['exists'] else 0
                status_code = 200 if db_result['exists'] else 404
                
                return jsonify({
                    "placa": license_plate,
                    "isOnDatabase": is_on_database,
                    "status": status_code,
                    "method": "gcp_vision"
                }), 200
            else:
                print(f"❌ No license plate detected with GCP Vision")
                return jsonify({
                    "placa": None,
                    "isOnDatabase": 0,
                    "status": 404,
                    "method": "gcp_vision"
                }), 200
            
        except Exception as e:
            print(f"❌ Controller error (v2): {e}")
            return jsonify({
                "placa": None, 
                "isOnDatabase": 0,
                "status": 500,
                "error": "Internal server error",
                "method": "gcp_vision"
            }), 200