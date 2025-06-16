import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseService:
    def __init__(self):
        self.connection_params = {
            'host': 'co8c1665c0p5k.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com',
            'database': 'd4eolt1svo0n1e',
            'user': 'u3c7qgf7k244nu',
            'password': 'p25a2499c11d9ffb0b80460a3f2339a6ad33454bf591899c47e5955c51e0dc3ac',
            'port': 5432
        }
    
    def get_connection(self):
        """Get database connection"""
        try:
            connection = psycopg2.connect(**self.connection_params)
            return connection
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def check_license_plate_exists(self, plate: str) -> Dict[str, Any]:
        """
        Check if license plate exists in vehiculos table
        
        Args:
            plate: License plate to check
            
        Returns:
            Dictionary with exists status and vehicle data if found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Query to check if plate exists
                    query = "SELECT * FROM vehiculos WHERE placa = %s"
                    cursor.execute(query, (plate,))
                    result = cursor.fetchone()
                    
                    if result:
                        logger.info(f"License plate {plate} found in database")
                        return {
                            'exists': True,
                            'vehicle_data': dict(result)
                        }
                    else:
                        logger.info(f"License plate {plate} not found in database")
                        return {
                            'exists': False,
                            'vehicle_data': None
                        }
                        
        except Exception as e:
            logger.error(f"Error checking license plate in database: {str(e)}")
            return {
                'exists': False,
                'vehicle_data': None,
                'error': str(e)
            }
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    logger.info("Database connection test successful")
                    return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False