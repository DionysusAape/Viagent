"""Logger configuration for the project"""
import os
import logging
from pathlib import Path
from datetime import datetime
from graph.schema import GraphState

class Logger:
    """Logger class"""
    def __init__(self, log_level: str = "INFO"):
        self.logdir = Path(__file__).parent.parent.parent / "logs"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("Viagent")
        self.logger.setLevel(log_level)

        formatter = logging.Formatter('%(levelname)s - %(message)s')
        self.logfile = self.logdir / f"viagent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.file_handler = logging.FileHandler(self.logfile)
        self.file_handler.setLevel(log_level)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(log_level)
        self.console_handler.setFormatter(formatter)
        self.logger.addHandler(self.console_handler)


    def info(self, message: str):
        """Log an info message"""
        self.logger.info(message)

    def debug(self, message: str):
        """Log a debug message"""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message"""
        self.logger.error(message)

    def log_agent_status(self, agent: str, object: str, status: str):
        """Log the status of an agent"""
        if object:
            msg = f"Agent:{agent} | Object:{object} | Status:{status}"
        else:
            msg = f"Agent:{agent} | Status:{status}"
        self.info(msg)

    def log_result(self, agent: str, object: str, result: str):
        """Log the result of an agent"""
        if object:
            msg = f"Agent:{agent} | Object:{object} | Result:{result}"
        else:
            msg = f"Agent:{agent} | Result:{result}"
        self.info(msg)

logger = Logger()