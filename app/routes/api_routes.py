from flask import Blueprint
from ..controllers.health_controller import HealthController
from ..controllers.license_plate_controller import LicensePlateController


def create_api_routes():
    api_bp = Blueprint('api', __name__)
    
    health_controller = HealthController()
    license_plate_controller = LicensePlateController()
    
    @api_bp.route('/health', methods=['GET'])
    def health():
        return health_controller.health_check()
    
    @api_bp.route('/detect-license-plate/v2', methods=['POST'])
    def detect_license_plate_v2():
        return license_plate_controller.detect_license_plate_v2()
    
    @api_bp.route('/detect-license-plate/v1', methods=['POST'])
    def detect_license_plate_v1():
        return license_plate_controller.detect_license_plate()
    
    return api_bp