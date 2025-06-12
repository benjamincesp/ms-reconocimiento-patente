from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
from skimage import io
from skimage.segmentation import clear_border
import re

app = Flask(__name__)

class LicensePlateRecognizer:
    def __init__(self):
        self.MIN_ASPECT_RATIO = 2
        self.MAX_ASPECT_RATIO = 6
        self.MIN_WIDTH = 100
        self.MIN_HEIGHT = 30
        
    def grayscale(self, img):
        """Convert image to grayscale"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def apply_threshold(self, img, threshold_value=170):
        """Apply binary threshold with inversion"""
        return cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
    
    def find_contours(self, img):
        """Find contours in the image"""
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def is_license_plate(self, contour):
        """Check if contour could be a license plate based on aspect ratio and size"""
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        return (self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO and 
                w >= self.MIN_WIDTH and h >= self.MIN_HEIGHT)
    
    def crop_license_plate(self, img, contour):
        """Crop the license plate region from the image"""
        x, y, w, h = cv2.boundingRect(contour)
        return img[y:y+h, x:x+w]
    
    def process_license_plate(self, license_plate_img):
        """Additional processing for the license plate region"""
        # Clear border to remove edge artifacts
        cleared = clear_border(license_plate_img)
        
        # Invert the image (make text black on white background)
        inverted = cv2.bitwise_not(cleared)
        
        return inverted
    
    def extract_text(self, processed_img):
        """Extract text from processed license plate image using Tesseract"""
        try:
            # Configure Tesseract for license plate recognition
            config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(processed_img, config=config)
            
            # Clean the extracted text
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
            
            return cleaned_text
        except Exception as e:
            print(f"Error in text extraction: {e}")
            return ""
    
    def validate_plate_format(self, text):
        """Validate if text matches Chilean license plate patterns"""
        if not text or len(text) < 5:
            return False
        
        # Chilean license plate patterns
        patterns = [
            r'^[A-Z]{4}[0-9]{2}$',  # AAAA##
            r'^[A-Z]{2}[0-9]{4}$',  # AA####
            r'^[A-Z]{3}[0-9]{3}$',  # AAA###
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        # Additional validation: at least 2 letters and 1 number
        letter_count = sum(1 for c in text if c.isalpha())
        number_count = sum(1 for c in text if c.isdigit())
        
        return letter_count >= 2 and number_count >= 1
    
    def recognize(self, img_bytes):
        """Main recognition pipeline"""
        try:
            # Decode image from bytes
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            print(f"Processing image of size: {img.shape}")
            
            # Step 1: Convert to grayscale
            gray_img = self.grayscale(img)
            print("✓ Converted to grayscale")
            
            # Step 2: Apply binary threshold with multiple values
            threshold_values = [170, 150, 190, 130, 210]
            
            for threshold in threshold_values:
                print(f"Trying threshold: {threshold}")
                
                # Apply threshold
                binary_img = self.apply_threshold(gray_img, threshold)
                
                # Step 3: Find contours
                contours = self.find_contours(binary_img)
                print(f"Found {len(contours)} contours")
                
                # Step 4: Filter contours for potential license plates
                license_plates = []
                for contour in contours:
                    if self.is_license_plate(contour):
                        license_plates.append(contour)
                
                print(f"Found {len(license_plates)} potential license plates")
                
                # Step 5: Process each potential license plate
                for i, plate_contour in enumerate(license_plates):
                    print(f"Processing plate candidate {i+1}")
                    
                    # Crop license plate region
                    license_plate_img = self.crop_license_plate(gray_img, plate_contour)
                    
                    if license_plate_img.size == 0:
                        continue
                    
                    # Additional processing
                    processed_img = self.process_license_plate(license_plate_img)
                    
                    # Extract text
                    extracted_text = self.extract_text(processed_img)
                    print(f"Extracted text: '{extracted_text}'")
                    
                    # Validate format
                    if self.validate_plate_format(extracted_text):
                        print(f"✓ Valid plate found: {extracted_text}")
                        return extracted_text
                    
                    # Try different processing approaches
                    # Try without clearing border
                    inverted_only = cv2.bitwise_not(license_plate_img)
                    text2 = self.extract_text(inverted_only)
                    print(f"Alternative extraction: '{text2}'")
                    
                    if self.validate_plate_format(text2):
                        print(f"✓ Valid plate found (alternative): {text2}")
                        return text2
                    
                    # Try with original orientation
                    text3 = self.extract_text(license_plate_img)
                    print(f"Original orientation: '{text3}'")
                    
                    if self.validate_plate_format(text3):
                        print(f"✓ Valid plate found (original): {text3}")
                        return text3
            
            print("No valid license plate found")
            return None
            
        except Exception as e:
            print(f"Error in recognition pipeline: {e}")
            return None

# Initialize the recognizer
lpr = LicensePlateRecognizer()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "service": "license-plate-recognition",
        "algorithm": "santifiorino-lpr"
    }), 200

@app.route('/detect-license-plate', methods=['POST'])
@app.route('/detect-license-plate/v1', methods=['POST'])
def detect_license_plate():
    """License plate detection endpoint"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({"placa": None, "error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"placa": None, "error": "Empty filename"}), 400
        
        # Validate file format
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"placa": None, "error": "Invalid file format"}), 400
        
        # Read image bytes
        image_bytes = file.read()
        print(f"\nProcessing image: {file.filename}")
        
        # Recognize license plate
        detected_plate = lpr.recognize(image_bytes)
        
        if detected_plate:
            print(f"✅ Detection successful: {detected_plate}")
            return jsonify({"placa": detected_plate}), 200
        else:
            print("❌ No license plate detected")
            return jsonify({"placa": None}), 404
        
    except Exception as e:
        print(f"❌ Endpoint error: {e}")
        return jsonify({"placa": None, "error": "Internal server error"}), 500

if __name__ == '__main__':
    # Check if Tesseract is available
    try:
        pytesseract.get_tesseract_version()
        print("✓ Tesseract OCR available")
        print("✓ Starting License Plate Recognition Service")
        print("✓ Algorithm: Based on santifiorino/license-plate-recognition")
        print("="*50)
    except Exception as e:
        print(f"❌ Tesseract not available: {e}")
        print("Install Tesseract:")
        print("  Mac: brew install tesseract")
        print("  Ubuntu: sudo apt install tesseract-ocr")
        print("  Windows: Download from UB-Mannheim/tesseract")
    
    app.run(host='0.0.0.0', port=5001, debug=True)