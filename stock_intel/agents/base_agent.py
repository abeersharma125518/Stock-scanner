import abc
import datetime
import logging
from typing import Dict, List, Optional, Any
from stock_intel.db.database import DatabaseManager

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        self.db = db
        self.config = config or {}
        self.name = self.__class__.__name__
        self.results: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abc.abstractmethod
    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def validate(self) -> bool:
        pass

    def get_results(self) -> Dict[str, Any]:
        return self.results

    def log_start(self):
        self.logger.info(f"[{self.name}] Starting execution at {datetime.datetime.now()}")

    def log_end(self):
        self.logger.info(f"[{self.name}] Finished execution at {datetime.datetime.now()}")

    def save_state(self, key: str, data: Any):
        self.results[key] = data

    def load_state(self, key: str, default: Any = None) -> Any:
        return self.results.get(key, default)
