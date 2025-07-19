"""
Lightweight utility for timing code execution.

This module provides the Timer class for measuring elapsed time in a manual
or context-managed manner.
"""

import time
from typing import Optional


class Timer:
    """
    Utility class for measuring elapsed time.

    Can be used manually via `start()`/`end()` or as a context manager.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self.elapsed_time: float = 0.0

    def start(self):
        """Start the timer."""
        self._start_time = time.time()

    def end(self):
        """
        Stop the timer and record the elapsed time.

        Raises:
            RuntimeError: If the timer was not started.
        """
        if self._start_time is None:
            raise RuntimeError("Timer has not been started.")
        self.elapsed_time = time.time() - self._start_time
        self._start_time = None

    def get_time_str(self) -> str:
        """
        Get the elapsed time as a formatted string (HH:MM:SS).

        Returns:
            str: Elapsed time formatted as hours, minutes, and seconds.
        """
        return time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()
