import os


class Config:
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5001))
    
    # Environment detection
    IS_HEROKU = 'DYNO' in os.environ
    
    
class OCRConfig:
    TESSERACT_CMD = os.getenv('TESSERACT_CMD', None)
    
    
class ImageConfig:
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    
class AppInfo:
    VERSION = "1.0.0"
    SERVICE_NAME = "license-plate-recognition"
    ALGORITHM = "santifiorino-lpr"