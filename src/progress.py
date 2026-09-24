"""Small stderr-only progress reporters for public command-line workflows."""

from __future__ import print_function

import math
import sys
import time


def _elapsed(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "%dh%02dm%02ds" % (hours, minutes, seconds)
    if minutes:
        return "%dm%02ds" % (minutes, seconds)
    return "%ds" % seconds


def progress_message(message, stream=None):
    stream = stream or sys.stderr
    stream.write(message + "\n")
    stream.flush()


class StageProgress(object):
    """Report a short requested-stage list and each stage transition."""

    def __init__(self, title, stages, stream=None, clock=None):
        self.title = title
        self.stages = list(stages)
        self.stream = stream or sys.stderr
        self.clock = clock or time.monotonic
        self.started = {}
        self.workflow_started = self.clock()
        progress_message(title, self.stream)
        progress_message(
            "Requested stages (%d): %s" % (len(self.stages), " -> ".join(label for _key, label in self.stages)),
            self.stream,
        )

    def _position(self, key):
        for index, (candidate, label) in enumerate(self.stages, start=1):
            if candidate == key:
                return index, label
        raise ValueError("stage was not requested: %s" % key)

    def start(self, key):
        index, label = self._position(key)
        self.started[key] = self.clock()
        progress_message("-> [%d/%d] %s" % (index, len(self.stages), label), self.stream)

    def complete(self, key):
        index, label = self._position(key)
        elapsed = self.clock() - self.started.get(key, self.workflow_started)
        progress_message(
            "OK [%d/%d] %s — complete (%s)" % (index, len(self.stages), label, _elapsed(elapsed)),
            self.stream,
        )

    def fail(self, key, error):
        index, label = self._position(key)
        progress_message(
            "FAIL [%d/%d] %s — %s" % (index, len(self.stages), label, error),
            self.stream,
        )

    def finish(self):
        progress_message(
            "OK %s — %d/%d requested stages complete (%s)"
            % (self.title, len(self.stages), len(self.stages), _elapsed(self.clock() - self.workflow_started)),
            self.stream,
        )


class LoopProgress(object):
    """Show every update in a TTY and sparse milestones in redirected logs."""

    def __init__(self, label, total, stream=None, clock=None):
        self.label = label
        self.total = int(total)
        if self.total < 1:
            raise ValueError("progress total must be positive")
        self.stream = stream or sys.stderr
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interval = max(1, int(math.ceil(self.total / 10.0)))
        self.next_milestone = self.interval
        self.last = 0
        self._emit(0, transient=False)

    def _line(self, completed):
        percent = 100.0 * float(completed) / float(self.total)
        return "%s: %d/%d (%.1f%%), elapsed %s" % (
            self.label,
            completed,
            self.total,
            percent,
            _elapsed(self.clock() - self.started),
        )

    def _emit(self, completed, transient):
        if transient:
            self.stream.write("\r" + self._line(completed))
        else:
            self.stream.write(self._line(completed) + "\n")
        self.stream.flush()

    def update(self, completed):
        completed = max(self.last, min(int(completed), self.total))
        self.last = completed
        if self.interactive:
            self._emit(completed, transient=completed < self.total)
            return
        if completed == 1 or completed == self.total or completed >= self.next_milestone:
            self._emit(completed, transient=False)
            while self.next_milestone <= completed:
                self.next_milestone += self.interval

    def complete(self):
        if self.last < self.total:
            self.update(self.total)
        progress_message(
            "OK %s — %d/%d complete (%s)" % (
                self.label,
                self.total,
                self.total,
                _elapsed(self.clock() - self.started),
            ),
            self.stream,
        )


def cache_reused(label, stream=None):
    progress_message("OK %s — validated cache reused" % label, stream)


def run_stage(progress, key, operation):
    """Run one existing operation while preserving its result and exceptions."""
    if progress is None:
        return operation()
    progress.start(key)
    if key == "downstream":
        progress_message(
            "   Fitting, cross-validation and evaluation are running; "
            "this stage may take several minutes without intermediate output.",
            progress.stream,
        )
    try:
        result = operation()
    except Exception as error:
        progress.fail(key, str(error))
        raise
    progress.complete(key)
    return result
