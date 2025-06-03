import logging
import traceback
from colorlog import StreamHandler, ColoredFormatter


class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra):
        super().__init__(logger, extra)
        self.logger = logger
        self.extra = extra

    def log(self, level, msg, *args, **kwargs):
        if kwargs.get("exc_info"):
            # Format exception with full traceback
            exc_info = kwargs["exc_info"]
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            if exc_info:
                msg += "\n" + "".join(
                    traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])
                )

        if kwargs.get("extra", {}).get("formatter"):
            self.logger.handlers[0].setFormatter(kwargs["extra"]["formatter"])

        self.logger.log(level, msg, *args, **kwargs)


class_color_map = {
    "agent": {
        "INFO": "light_blue",
        "DEBUG": "cyan",
        "ERROR": "red",
        "WARNING": "yellow",
        "CRITICAL": "red,bg_white",
    },
    "tool": {
        "INFO": "yellow",
        "DEBUG": "cyan",
        "ERROR": "red",
        "WARNING": "yellow",
        "CRITICAL": "red,bg_white",
    },
    "agent_response": {
        "INFO": "green",
        "DEBUG": "cyan",
        "ERROR": "red",
        "WARNING": "yellow",
        "CRITICAL": "red,bg_white",
    },
}

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

# Create console handler with detailed formatting
handler = StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter that includes timestamp and stack info for errors
formatter = ColoredFormatter(
    "%(asctime)s %(log_color)s%(class_name)s:%(levelname)s%(reset)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "light_blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)

handler.setFormatter(formatter)
logger.addHandler(handler)

# Prevent logs from propagating to the root logger
logger.propagate = False
