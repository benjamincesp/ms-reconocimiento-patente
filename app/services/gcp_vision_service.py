import base64
import re
from typing import Optional, List, Dict, Any
from google.cloud import vision
from google.oauth2 import service_account
import json
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

class GCPVisionService:
    def __init__(self):
        self.credentials_path = "app/config/credentials/copec-462414-cad89218ad0f.json"
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> vision.ImageAnnotatorClient:
        """Initialize Google Cloud Vision client with service account"""
        try:
            # Check if credentials file exists
            if os.path.exists(self.credentials_path):
                # Initialize client with service account credentials file
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                client = vision.ImageAnnotatorClient(credentials=credentials)
                logger.info("GCP Vision client initialized successfully with service account file")
            else:
                # Fallback to environment variables or default credentials
                # Set GOOGLE_APPLICATION_CREDENTIALS environment variable
                client = vision.ImageAnnotatorClient()
                logger.info("GCP Vision client initialized with default credentials")
            
            return client
        except Exception as e:
            logger.error(f"Failed to initialize GCP Vision client: {str(e)}")
            raise
    
    def extract_text_from_image(self, image_data: bytes) -> Optional[str]:
        """
        Extract text from image using Google Cloud Vision API
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            # Create Vision API image object
            image = vision.Image(content=image_data)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if response.error.message:
                logger.error(f"GCP Vision API error: {response.error.message}")
                return None
            
            if not texts:
                logger.warning("No text detected in image")
                return None
            
            # Return the first (most comprehensive) text annotation
            extracted_text = texts[0].description
            logger.info(f"Text extracted successfully: {extracted_text[:50]}...")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return None
    
    def extract_license_plate_from_image(self, image_data: bytes) -> Optional[str]:
        """
        Extract Chilean license plate from image using GCP Vision
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            License plate text or None if not found
        """
        try:
            extracted_text = self.extract_text_from_image(image_data)
            
            if not extracted_text:
                return None
            
            # Find Chilean license plate pattern in extracted text
            license_plate = self._extract_chilean_license_plate(extracted_text)
            
            if license_plate:
                logger.info(f"Chilean license plate detected: {license_plate}")
                return license_plate
            else:
                logger.warning("No valid Chilean license plate pattern found in extracted text")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting license plate: {str(e)}")
            return None
    
    def _extract_chilean_license_plate(self, text: str) -> Optional[str]:
        """
        Extract Chilean license plate from text using regex patterns
        
        Args:
            text: Text extracted from image
            
        Returns:
            License plate string or None if not found
        """
        try:
            # Clean text: remove extra spaces and newlines
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            
            # Chilean license plate patterns
            patterns = [
                # Current format with spaces: LL DD-NN (like "JG DJ-66")
                r'[A-Z]{2}\s+[A-Z]{2}-\d{2}',
                # Current format: LL-NN-NN (2 letters + hyphen + 2 numbers + hyphen + 2 numbers)
                r'[A-Z]{2}-\d{2}-\d{2}',
                # New format: LLLL·NN (4 letters + dot + 2 numbers)
                r'[A-Z]{4}[·•.]\d{2}',
                # Old format: LL·NNNN (2 letters + dot + 4 numbers)  
                r'[A-Z]{2}[·•.]\d{4}',
                # Alternative formats without separators
                r'[A-Z]{4}\s*\d{2}',
                r'[A-Z]{2}\s*\d{4}',
                # With spaces between characters
                r'[A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*[·•.]?\s*\d\s*\d',
                r'[A-Z]\s*[A-Z]\s*[·•.]?\s*\d\s*\d\s*\d\s*\d',
                # With hyphens and spaces (flexible)
                r'[A-Z]\s*[A-Z]\s*-?\s*\d\s*\d\s*-?\s*\d\s*\d',
                # More flexible pattern for spaced format
                r'[A-Z]{2}\s+[A-Z]{2}\s*-\s*\d{2}'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
                if matches:
                    # Clean the match: remove extra spaces and normalize
                    license_plate = matches[0].upper()
                    license_plate = re.sub(r'\s+', '', license_plate)
                    
                    # Normalize dot/bullet characters
                    license_plate = re.sub(r'[·•]', '.', license_plate)
                    
                    # Validate format
                    if self._validate_chilean_license_plate(license_plate):
                        # Remove hyphens and spaces from final result
                        license_plate = license_plate.replace('-', '').replace(' ', '')
                        return license_plate
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing license plate text: {str(e)}")
            return None
    
    def _validate_chilean_license_plate(self, plate: str) -> bool:
        """
        Validate Chilean license plate format
        
        Args:
            plate: License plate string
            
        Returns:
            True if valid Chilean format, False otherwise
        """
        try:
            # Current format with spaces: LL DD-NN (like "JG DJ-66")
            if re.match(r'^[A-Z]{2}\s+[A-Z]{2}-\d{2}$', plate):
                return True
            
            # Current format: LL-NN-NN (2 letters + hyphen + 2 numbers + hyphen + 2 numbers)
            if re.match(r'^[A-Z]{2}-\d{2}-\d{2}$', plate):
                return True
            
            # Remove dots, hyphens and spaces for other validations
            clean_plate = plate.replace('.', '').replace('-', '').replace(' ', '')
            
            # New format: 4 letters + 2 numbers
            if len(clean_plate) == 6 and clean_plate[:4].isalpha() and clean_plate[4:].isdigit():
                return True
            
            # Old format: 2 letters + 4 numbers
            if len(clean_plate) == 6 and clean_plate[:2].isalpha() and clean_plate[2:].isdigit():
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating license plate: {str(e)}")
            return False
    
    def get_detailed_text_analysis(self, image_data: bytes) -> Dict[str, Any]:
        """
        Get detailed text analysis including bounding boxes and confidence
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Dictionary with detailed analysis results
        """
        try:
            image = vision.Image(content=image_data)
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if response.error.message:
                logger.error(f"GCP Vision API error: {response.error.message}")
                return {"error": response.error.message}
            
            result = {
                "full_text": texts[0].description if texts else "",
                "text_blocks": [],
                "license_plate": None
            }
            
            # Process individual text blocks
            for text in texts[1:]:  # Skip first element (full text)
                vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                result["text_blocks"].append({
                    "text": text.description,
                    "bounding_box": vertices
                })
            
            # Extract license plate
            if result["full_text"]:
                result["license_plate"] = self._extract_chilean_license_plate(result["full_text"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detailed text analysis: {str(e)}")
            return {"error": str(e)}