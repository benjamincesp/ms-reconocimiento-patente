import logging
import sys
from datetime import datetime


class AppLogger:
    
    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @staticmethod
    def log_processing_step(step: str, details: str = ""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if details:
            print(f"[{timestamp}] {step}: {details}")
        else:
            print(f"[{timestamp}] {step}")
    
    @staticmethod
    def log_success(message: str):
        print(f"✅ {message}")
    
    @staticmethod
    def log_error(message: str):
        print(f"❌ {message}")
    
    @staticmethod
    def log_warning(message: str):
        print(f"⚠️  {message}")


def get_logger(name: str):
    """Get a logger instance for the given name"""
    return AppLogger.setup_logger(name)