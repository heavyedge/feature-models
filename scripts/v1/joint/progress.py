"""Logging helpers shared by the joint command-line scripts."""

import logging
import time


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class ProgressLogger:
    """Log percentage, elapsed time, and ETA without flooding stderr."""

    def __init__(self, label, total, percentage_interval=5):
        self.label = label
        self.total = total
        self.percentage_interval = percentage_interval
        self.started = time.monotonic()
        self.next_percentage = 0

    def update(self, completed):
        percentage = 100.0 if self.total == 0 else 100.0 * completed / self.total
        if percentage < self.next_percentage and completed < self.total:
            return

        elapsed = time.monotonic() - self.started
        if completed and completed < self.total:
            eta = elapsed * (self.total - completed) / completed
            timing = f"elapsed {elapsed:.1f}s, ETA {eta:.1f}s"
        else:
            timing = f"elapsed {elapsed:.1f}s"
        logging.info(
            "%s: %s/%s (%.1f%%), %s",
            self.label,
            f"{completed:,}",
            f"{self.total:,}",
            percentage,
            timing,
        )
        self.next_percentage = (
            int(percentage // self.percentage_interval) + 1
        ) * self.percentage_interval
