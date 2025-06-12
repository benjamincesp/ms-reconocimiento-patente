from flask import Flask, request, jsonify
import cv2
import numpy as np
import re
import io
import pytesseract
from PIL import Image

app = Flask(__name__)

def preprocess_image(image):
    """Preprocesa la imagen para mejorar la detección de placas"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro bilateral para reducir ruido manteniendo bordes
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Detectar bordes usando Canny
    edges = cv2.Canny(bilateral, 30, 200)
    
    return gray, edges

def find_license_plate_contours(edges, original_image):
    """Encuentra contornos que podrían ser placas de vehículos"""
    # Encontrar contornos
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordenar contornos por área (de mayor a menor)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    license_plate_contour = None
    
    for contour in contours:
        # Aproximar el contorno
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        # Si el contorno tiene 4 vértices, podría ser una placa
        if len(approx) == 4:
            license_plate_contour = approx
            break
    
    return license_plate_contour

def extract_text_from_plate(image, contour):
    """Extrae texto de la región de la placa"""
    if contour is None:
        return None
    
    # Crear máscara para la región de la placa
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Aplicar la máscara
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Encontrar la región de interés
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    
    # Recortar la imagen
    cropped = image[topx:bottomx+1, topy:bottomy+1]
    
    if cropped.size == 0:
        return None
    
    # Redimensionar para mejor OCR
    if cropped.shape[0] < 50 or cropped.shape[1] < 150:
        scale_factor = max(50 / cropped.shape[0], 150 / cropped.shape[1])
        new_width = int(cropped.shape[1] * scale_factor)
        new_height = int(cropped.shape[0] * scale_factor)
        cropped = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Aplicar OCR con Tesseract
    try:
        # Configuración optimizada para placas
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(cropped, config=custom_config)
        
        # Limpiar el texto
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        
        # Validar que tenga al menos 5 caracteres
        if len(cleaned_text) >= 5:
            return cleaned_text
        
        # Intentar con configuración alternativa
        custom_config2 = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text2 = pytesseract.image_to_string(cropped, config=custom_config2)
        cleaned_text2 = re.sub(r'[^A-Z0-9]', '', text2.upper().strip())
        
        if len(cleaned_text2) >= 5:
            return cleaned_text2
            
    except Exception as e:
        print(f"Error en OCR: {e}")
        return None
    
    return None

def validate_chilean_plate(plate_text):
    """Valida si el texto parece una placa chilena"""
    if not plate_text or len(plate_text) < 5:
        return False
    
    # Patrones típicos chilenos
    patterns = [
        r'^[A-Z]{4}[0-9]{2}$',  # AAAA##
        r'^[A-Z]{2}[0-9]{4}$',  # AA####
        r'^[A-Z]{3}[0-9]{3}$',  # AAA###
    ]
    
    for pattern in patterns:
        if re.match(pattern, plate_text):
            return True
    
    # Validación adicional: al menos 2 letras y 1 número
    letter_count = sum(1 for c in plate_text if c.isalpha())
    number_count = sum(1 for c in plate_text if c.isdigit())
    
    return letter_count >= 2 and number_count >= 1

def detect_license_plate(img_bytes):
    """Función principal para detectar placas"""
    try:
        # Decodificar imagen
        img_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        # Redimensionar si es muy grande
        height, width = image.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Preprocesar imagen
        gray, edges = preprocess_image(image)
        
        # Encontrar contornos de placas
        plate_contour = find_license_plate_contours(edges, image)
        
        # Extraer texto de la placa
        if plate_contour is not None:
            plate_text = extract_text_from_plate(gray, plate_contour)
            
            if plate_text and validate_chilean_plate(plate_text):
                return plate_text
        
        # Si no se encuentra por contornos, intentar OCR directo
        try:
            # Mejorar imagen completa
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR en imagen completa
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            full_text = pytesseract.image_to_string(enhanced, config=custom_config)
            
            # Buscar patrones de placa en el texto completo
            words = full_text.split()
            for word in words:
                cleaned = re.sub(r'[^A-Z0-9]', '', word.upper())
                if validate_chilean_plate(cleaned):
                    return cleaned
                    
        except Exception as e:
            print(f"Error en OCR directo: {e}")
        
        return None
        
    except Exception as e:
        print(f"Error en detección: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "service": "license-plate-detection"
    }), 200

@app.route('/detect-license-plate', methods=['POST'])
def detect_license_plate_endpoint():
    """Endpoint principal para detectar placas"""
    try:
        # Validar que se envió una imagen
        if 'image' not in request.files:
            return jsonify({"placa": None, "error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"placa": None, "error": "Empty filename"}), 400
        
        # Validar formato de archivo
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"placa": None, "error": "Invalid file format"}), 400
        
        # Leer imagen
        image_bytes = file.read()
        
        # Detectar placa
        detected_plate = detect_license_plate(image_bytes)
        
        if detected_plate:
            return jsonify({"placa": detected_plate}), 200
        else:
            return jsonify({"placa": None}), 404
        
    except Exception as e:
        print(f"Error en endpoint: {e}")
        return jsonify({"placa": None, "error": "Internal server error"}), 500

if __name__ == '__main__':
    # Verificar que Tesseract esté instalado
    try:
        pytesseract.get_tesseract_version()
        print("Tesseract OCR disponible")
    except Exception as e:
        print(f"ADVERTENCIA: Tesseract no disponible - {e}")
        print("Instala Tesseract: brew install tesseract (Mac) o apt install tesseract-ocr (Ubuntu)")
    
    app.run(host='0.0.0.0', port=5001, debug=True)