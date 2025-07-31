from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """Base class for all data collectors"""
    
    def __init__(self, name: str, rate_limit: float = 1.0):
        self.name = name
        self.rate_limit = rate_limit
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            logger.debug(f"Rate limiting {self.name}: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @abstractmethod
    def collect_data(self, **kwargs) -> List[Any]:
        """Collect data - must be implemented by subclasses"""
        pass