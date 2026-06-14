from __future__ import annotations

import time

from .models import Timer


def timer_start(timer: Timer) -> None:
    timer.count_start = time.perf_counter_ns()


def timer_stop(timer: Timer) -> None:
    timer.count_stop = time.perf_counter_ns()


def timer_time(timer: Timer) -> float:
    return (timer.count_stop - timer.count_start) / 1_000_000_000.0
