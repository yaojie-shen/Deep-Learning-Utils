# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 13:41
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : timer.py

import datetime
import time
from collections import defaultdict, deque
from typing import Optional

from tabulate import tabulate


def get_timestamp() -> str:
    """Return the current time in a format suitable for filenames.

    Examples:
        >>> get_timestamp()
        '2022-10-11T13-41-45W'
    """
    return "{0:%Y-%m-%dT%H-%M-%SW}".format(datetime.datetime.now())


def get_readable_timestamp():
    """Return the current time in a readable format.

    Examples:
        >>> get_readable_timestamp()
        '2022-10-11 13:41:45'
    """
    return "{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())


def get_current_time_in_ms(precision: int) -> float:
    """Return the current time in milliseconds."""
    return round(time.time() * 1000, precision)


class Timer:
    """A simple timer for getting duration in milliseconds."""

    def __init__(self, precision=3):
        assert precision >= 0, "precision must be greater than or equal to 0"
        self._precision = precision
        self._start_time = self.get_current_time_in_ms()

    def get_current_time_in_ms(self) -> float:
        """Return the current time in milliseconds."""
        return get_current_time_in_ms(self._precision)

    def reset(self) -> float:
        self._start_time = self.get_current_time_in_ms()
        return self._start_time

    def get_duration(self) -> float:
        return self.get_current_time_in_ms() - self._start_time

    def get_duration_and_reset(self) -> float:
        duration = self.get_duration()
        self.reset()
        return duration


class ExecutionTimer(Timer):
    """A timer for tracking the execution time of sequential stages.

    Args:
        history_size: Number of history to store for each stage. If None, all history will be stored.
        precision: Precision of the duration in milliseconds.
        start_prompt: Format string for the start prompt. If None, the default prompt will be used.
        end_prompt: Format string for the end prompt. If None, the default prompt will be used.
        log: Whether to print the start and end prompts at the beginning and end of each stage.
        name: Recognizable name of the timer to be used in outputs.

    Examples:
        >>> timer = ExecutionTimer(log=True)
        >>> timer.start_stage("stage_1")
        (2025-10-21 16:17:32) => Starting stage: stage_1...
        >>> time.sleep(0.1) # do something
        >>> timer.start_stage("stage_2")
        (2025-10-21 16:17:32) => Finished stage: stage_1 | Took 103.030 ms.
        (2025-10-21 16:17:32) => Starting stage: stage_2...
        >>> time.sleep(0.2) # do something
        >>> timer.end_stage("stage_2")
        (2025-10-21 16:17:32) => Finished stage: stage_2 | Took 203.100 ms.
        >>> timer.print_table()
        Stage      Total (ms)    Count    Min (ms)    Max (ms)    Avg (ms)
        -------  ------------  -------  ----------  ----------  ----------
        stage_1        103.03        1      103.03      103.03      103.03
        stage_2        203.10        1      203.10      203.10      203.10
        -------------------------
        Total Time: 306.13 ms
    """

    def __init__(
        self,
        history_size: Optional[int] = None,
        precision: int = 2,
        start_prompt: str = None,
        end_prompt: str = None,
        log: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(precision=precision)

        self._stage_name = None

        self._stage_log = defaultdict(lambda: deque(maxlen=history_size))

        self._enable_log = log
        self._start_log_prompt = (
            "({ctime}) {name} => starting stage: {stage}..."
            if start_prompt is None
            else start_prompt
        )
        self._end_log_prompt = (
            "({ctime}) {name} => finished stage: {stage} | took {duration:.3f} ms."
            if end_prompt is None
            else end_prompt
        )

        self._name = name

    def start_stage(self, name: Optional[str] = None):
        """
        Log the start of a stage. If there is a previous stage, the previous stage will be ended automatically by
        calling `end_stage()`.

        Args:
            name: The name of the stage to start. If None, one must be specified in the next call to `end_stage`.

        Note:
            Only for sequential use.
        """
        assert name is None or isinstance(name, str), f"Invalid stage name: {name}"

        # Previous stage is not finished
        if self._stage_name is not None:
            self.end_stage()  # End previous stage

        # Set current stage name
        self._stage_name = name
        self.reset()

        if self._enable_log and name is not None:
            print(
                self._start_log_prompt.format(
                    ctime=get_readable_timestamp(),
                    stage=self._stage_name,
                    name=self._name or "ExecutionTimer",
                )
            )

    def end_stage(self, name: Optional[str] = None):
        """
        Log the end of a stage.

        Args:
            name: The name of the stage to end. If None, the name of the last call to `start_stage` must be specified.

        Note:
            Only for sequential use.
        """
        assert name is None or isinstance(name, str), f"Invalid stage name: {name}"
        assert name is not None or self._stage_name is not None, (
            "Stage name is unknown: it must be set once at the beginning or end of the stage"
        )

        if name is None:
            name = self._stage_name
        elif self._stage_name is not None:
            assert self._stage_name == name, (
                f"Stage name mismatch: {self._stage_name} (in process) != {name} (trying to end)"
            )

        duration = self.get_duration_and_reset()
        self._stage_log[name].append(duration)
        self._stage_name = None

        if self._enable_log:
            print(
                self._end_log_prompt.format(
                    ctime=get_readable_timestamp(),
                    stage=name,
                    duration=duration,
                    name=self._name or "ExecutionTimer",
                )
            )

    def __str__(self):
        out = ""
        info = self.summary()
        for stage_name, stage_info in info.items():
            out += (
                f"[{stage_name}]:\n\ttotal {stage_info['total']} ms\n"
                f"\t{stage_info['count']}-iters range ({stage_info['min']}, {stage_info['max']})\n"
                f"\tavg {stage_info['avg']} ms\n"
            )
        return out

    def summary(self):
        """Return a summary of all stages as a dictionary.

        Each key is a stage name, and its value is a dictionary with:

        - "total": total time (ms)
        - "avg": average time (ms)
        - "min": minimum time (ms)
        - "max": maximum time (ms)
        - "last": last recorded time (ms)
        - "count": number of runs
        """
        return {
            k: {
                "total": round(sum(v), self._precision),
                "avg": round(sum(v) / len(v), self._precision),
                "min": round(min(v), self._precision),
                "max": round(max(v), self._precision),
                "last": round(v[-1], self._precision),
                "count": len(v),
            }
            for k, v in self._stage_log.items()
        }

    def print_table(self):
        """Print the summary of the timer in a table format."""
        if self._stage_name is not None:
            self.end_stage()

        summary = self.summary()
        data = [
            [k, v["total"], v["count"], v["min"], v["max"], v["avg"], v["last"]]
            for k, v in summary.items()
        ]
        total_time = sum(v["total"] for v in summary.values())

        table_str = tabulate(
            data,
            headers=[
                "Stage",
                "Total (ms)",
                "Count",
                "Min (ms)",
                "Max (ms)",
                "Avg (ms)",
                "Last (ms)",
            ],
            tablefmt="pipe",
            floatfmt=f".0{self._precision}f",
        )

        if self._name is not None:
            name_str = (
                f"=> {self._name} | Total Time: {total_time:.{self._precision}f} ms"
            )
        else:
            name_str = f"=> Total Time: {total_time:.{self._precision}f} ms"

        print(f"{name_str}\n{table_str}")


__all__ = [
    "get_timestamp",
    "get_readable_timestamp",
    "get_current_time_in_ms",
    "Timer",
    "ExecutionTimer",
]
