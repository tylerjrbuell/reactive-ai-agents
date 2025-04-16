from typing import Any
from config.logging import logger, class_color_map, LoggerAdapter
from colorlog import ColoredFormatter, StreamHandler
import logging


class Logger:
    def __init__(self, name: str, type: str, level: str = "info"):
        self.name = name
        self.type = type

        # Create a custom logging formatter
        def get_color(type, level):
            color = class_color_map.get(type, {}).get(level, "white")
            return color

        colors = {
            "DEBUG": get_color(self.type, "DEBUG"),
            "INFO": get_color(self.type, "INFO"),
            "WARNING": get_color(self.type, "WARNING"),
            "ERROR": get_color(self.type, "ERROR"),
            "CRITICAL": get_color(self.type, "CRITICAL"),
        }
        self.formatter = ColoredFormatter(
            f"%(asctime)s %(log_color)s%(class_name)s:%(levelname)s%(reset)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=colors,
            reset=True,
        )

        logger.setLevel(getattr(logging, level.upper()))
        self.logger = LoggerAdapter(logger, {"class_name": self.name})

    def get_logger(self):
        return logger

    def log(self, message: str, level: str = "info", exc_info=None):
        logger.log(
            getattr(logging, level.upper()),
            f"{self.name}: {message}",
            extra={"class_name": self.name, "formatter": self.formatter},
            exc_info=exc_info,
        )

    def info(self, message: Any):
        self.logger.info(
            msg=message,
            extra={"class_name": self.name, "formatter": self.formatter},
        )

    def debug(self, message: str):
        self.logger.debug(
            msg=message, extra={"class_name": self.name, "formatter": self.formatter}
        )

    def error(
        self, message: str, exc_info=True
    ):  # Changed to include exc_info by default
        logger.error(
            msg=message,
            extra={"class_name": self.name, "formatter": self.formatter},
            exc_info=exc_info,
        )

    def warning(self, message: str, exc_info=None):
        logger.warning(
            msg=message,
            extra={"class_name": self.name, "formatter": self.formatter},
            exc_info=exc_info,
        )

    def critical(
        self, message: str, exc_info=True
    ):  # Changed to include exc_info by default
        logger.critical(
            msg=message,
            extra={"class_name": self.name, "formatter": self.formatter},
            exc_info=exc_info,
        )
