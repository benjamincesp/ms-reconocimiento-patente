from flask import Flask
from flask_cors import CORS
import pytesseract
from app.routes.api_routes import create_api_routes
from app.config.settings import Config, OCRConfig, AppInfo
from app.services.ocr_service import OCRService


def create_app():
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app, origins=["http://localhost:4200"])
    
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['DEBUG'] = Config.DEBUG
    
    if OCRConfig.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = OCRConfig.TESSERACT_CMD
    
    api_routes = create_api_routes()
    app.register_blueprint(api_routes)
    
    return app


def main():
    if OCRService.check_tesseract_availability():
        print("✓ Tesseract OCR available")
        print(f"✓ Starting {AppInfo.SERVICE_NAME}")
        print(f"✓ Algorithm: {AppInfo.ALGORITHM}")
        print(f"✓ Version: {AppInfo.VERSION}")
        print("="*50)
    else:
        print("❌ Tesseract not available")
        print("Install Tesseract:")
        print("  Mac: brew install tesseract")
        print("  Ubuntu: sudo apt install tesseract-ocr")
        print("  Windows: Download from UB-Mannheim/tesseract")
        print("="*50)
    
    app = create_app()
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)


if __name__ == '__main__':
    main()