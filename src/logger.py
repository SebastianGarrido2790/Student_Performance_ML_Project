import logging
import os
from datetime import datetime

# Create a logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Generate a log file name with a datetime stamp
log_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, log_filename)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(module)s.%(funcName)s - %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__ == "__main__":
#     logging.info("Logging has started")
