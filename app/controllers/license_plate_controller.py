from flask import request, jsonify
from ..services.license_plate_service import LicensePlateService
from ..services.gcp_vision_service import GCPVisionService
from ..utils.validators import FileValidator


class LicensePlateController:
    
    def __init__(self):
        self.license_plate_service = LicensePlateService()
        self.gcp_vision_service = GCPVisionService()
        self.file_validator = FileValidator()
    
    def detect_license_plate(self):
        try:
            validation_result = self.file_validator.validate_image_upload(request)
            if not validation_result['valid']:
                return jsonify({
                    "placa": None, 
                    "error": validation_result['error']
                }), validation_result['status_code']
            
            file = request.files['image']
            image_bytes = file.read()
            
            print(f"\nProcessing image: {file.filename}")
            
            result = self.license_plate_service.recognize(image_bytes)
            
            if result.plate:
                print(f"✅ Detection successful: {result.plate} (method: {result.processing_method})")
                return jsonify({"placa": result.plate}), 200
            else:
                print(f"❌ No license plate detected (method: {result.processing_method})")
                if result.error:
                    print(f"Error: {result.error}")
                return jsonify({"placa": None}), 404
            
        except Exception as e:
            print(f"❌ Controller error: {e}")
            return jsonify({"placa": None, "error": "Internal server error"}), 500
    
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
                return jsonify({
                    "placa": license_plate,
                    "method": "gcp_vision"
                }), 200
            else:
                print(f"❌ No license plate detected with GCP Vision")
                return jsonify({
                    "placa": None,
                    "method": "gcp_vision"
                }), 404
            
        except Exception as e:
            print(f"❌ Controller error (v2): {e}")
            return jsonify({
                "placa": None, 
                "error": "Internal server error",
                "method": "gcp_vision"
            }), 500