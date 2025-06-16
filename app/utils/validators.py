from flask import request
from ..config.settings import ImageConfig


class FileValidator:
    
    @staticmethod
    def validate_image_upload(request_obj):
        if 'image' not in request_obj.files:
            return {
                'valid': False,
                'error': 'No image provided',
                'status_code': 400
            }
        
        file = request_obj.files['image']
        
        if file.filename == '':
            return {
                'valid': False,
                'error': 'Empty filename',
                'status_code': 400
            }
        
        if not FileValidator._allowed_file(file.filename):
            return {
                'valid': False,
                'error': 'Invalid file format. Allowed: png, jpg, jpeg',
                'status_code': 400
            }
        
        return {
            'valid': True,
            'error': None,
            'status_code': 200
        }
    
    @staticmethod
    def _allowed_file(filename):
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in ImageConfig.ALLOWED_EXTENSIONS)