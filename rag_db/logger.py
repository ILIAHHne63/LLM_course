# rag_db/logger.py

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "build" / "rag_db.log"))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Чтобы не дублировались хендлеры при повторных импортам
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1) Вывод в консоль
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 2) Лог в файл
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
