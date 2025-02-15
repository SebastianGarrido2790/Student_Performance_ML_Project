import sys
from src.logger import logging


def error_message_detail(error: Exception, error_detail) -> str:
    """
    Constructs a detailed error message using the exception and its traceback information.

    Args:
        error (Exception): The exception instance.
        error_detail: An object with an exc_info() method (typically the sys module).

    Returns:
        str: A formatted error message including the filename and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    error_message = f"Error occurred in python script [{file_name}] at line [{line_number}]: {str(error)}"
    return error_message


class CustomException(Exception):
    """
    Custom Exception that enriches the error message with detailed traceback information.
    """

    def __init__(self, error: Exception, error_detail) -> None:
        """
        Initializes the CustomException with a detailed error message.

        Args:
            error (Exception): The original exception.
            error_detail: An object with an exc_info() method (typically the sys module).
        """
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        return self.error_message


# Quick testing:
# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.info("Divide by Zero encountered.")
#         raise CustomException(e, sys)
