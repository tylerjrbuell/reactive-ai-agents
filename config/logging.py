import logging
from colorlog import StreamHandler


class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra):
        super().__init__(logger, extra)
        self.logger = logger
        self.extra = extra

    def process(self, msg, kwargs):
        (
            self.logger.handlers[0].setFormatter(kwargs["extra"].get("formatter"))
            if kwargs["extra"].get("formatter")
            else None
        )
        return msg, kwargs

    def log(self, level, msg, *args, **kwargs):
        (
            self.logger.handlers[0].setFormatter(kwargs["extra"].get("formatter"))
            if kwargs["extra"].get("formatter")
            else None
        )
        self.logger.log(level, msg, *args, **kwargs)


class_color_map = {
    "agent": {"INFO": "light_blue", "DEBUG": "cyan", "ERROR": "red"},
    "tool": {"INFO": "yellow", "DEBUG": "cyan", "ERROR": "red"},
    "agent_response": {"INFO": "green", "DEBUG": "cyan", "ERROR": "red"},
}


logger = logging.getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)
