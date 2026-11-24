"""Application-level progress watchdog and a test-only fault hook.

c10d/NCCL timeouts were measured not to end a job after one rank failed
(43,008 atoms on 2 GPUs: OOM, then a silent 480 s hang). This daemon thread
watches host progress only: the driver enters a phase before each legitimately
long section and beats after every completed step. On expiry it dumps every
thread's stack and takes launch.fail_fast(). It never calls into CUDA, so it
cannot invalidate a concurrent global-mode graph capture. Each poll also
looks for another rank's fail_fast() announcement (launch.PeerFailureProbe):
a rank waiting in NCCL for a dead peer then exits within one poll instead of
at its deadline, whatever the launcher.
"""
from __future__ import annotations

import faulthandler
from functools import lru_cache
import os
import sys
import threading
import time
import traceback

from . import launch

_active = None


class ProgressWatchdog:
    """Phase-aware progress deadline; ``seconds=0`` tracks phases without a thread.

    phase(name) grants ``seconds`` (initialization, oracle, capture, teardown).
    beat() after a completed step grants max(step_seconds, 20 x its wall time).
    If this thread cannot run because a hang holds the GIL, a C-level
    faulthandler timer exits ``backstop_seconds`` after the deadline instead.
    Every poll (``poll_seconds``) also checks for a peer's fail_fast()
    announcement and then fails fast with launch.PeerFailedError; a disabled
    watchdog (``seconds=0``) has no thread and does neither.
    """

    STEP_FACTOR = 20.0

    def __init__(self, seconds: float = 600.0, *, step_seconds: float = 60.0, rank: int = 0,
                 output=None, poll_seconds: float = 1.0, backstop_seconds: float = 60.0,
                 on_expire=None, clock=time.monotonic):
        if seconds < 0 or min(step_seconds, poll_seconds, backstop_seconds) <= 0:
            raise ValueError("watchdog seconds must be nonnegative and its periods positive")
        self.seconds, self.step_seconds = float(seconds), float(step_seconds)
        self.rank, self.output = rank, output
        self.poll_seconds, self.backstop_seconds = poll_seconds, backstop_seconds
        self._on_expire, self._clock = on_expire, clock
        self._peers = launch.PeerFailureProbe(rank)
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = None
        self.phase_name = "startup"
        self._step_started = clock()
        self._span, self._deadline = self.seconds, self._step_started + self.seconds

    @property
    def enabled(self) -> bool:
        return self.seconds > 0

    def start(self) -> "ProgressWatchdog":
        global _active
        if _active is not None:
            raise RuntimeError("a progress watchdog is already active in this process")
        _active = self
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="matris-progress-watchdog",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        global _active
        if _active is self:
            _active = None
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=10 * self.poll_seconds)
            self._thread = None

    def phase(self, name: str, *, long: bool = True) -> None:
        """Start a named phase; if ``long`` its first step may take up to ``seconds``.

        ``long=False`` only relabels, keeping the steady-state step deadline.
        """
        if long:
            self._grant(name, self.seconds, restart_step=True)
        else:
            with self._lock:
                self.phase_name = name

    def beat(self) -> None:
        """A step completed; the next one may take max(step_seconds, 20x this one)."""
        with self._lock:
            now = self._clock()
            span = max(self.step_seconds, self.STEP_FACTOR * (now - self._step_started))
            self._step_started, self._span, self._deadline = now, span, now + span

    def enter(self, name: str):
        """Nested long section inside a step; pass the token back to resume().

        The section gets ``seconds`` and is excluded from the step time that
        sets the next step deadline (a recapture must not stretch it 20x).
        """
        with self._lock:
            token = (self.phase_name, self._span, self._step_started, self._clock())
        self._grant(name, self.seconds)
        return token

    def resume(self, token) -> None:
        name, span, step_started, entered = token
        with self._lock:
            now = self._clock()
            # Shift the step start past the section; nested sections are not
            # subtracted twice because the outer token overwrites the inner.
            self._step_started = step_started + (now - entered)
            self.phase_name, self._span, self._deadline = name, span, now + span

    def _grant(self, name, span, *, restart_step=False):
        with self._lock:
            now = self._clock()
            if restart_step:
                self._step_started = now
            self.phase_name, self._span, self._deadline = name, span, now + span

    def check(self) -> str | None:
        """Expiry message, or None while progress is within the deadline."""
        with self._lock:
            if not self.enabled or self._clock() <= self._deadline:
                return None
            return f"no progress within {self._span:.1f} s in phase {self.phase_name!r}"

    def _run(self):
        armed = None
        try:
            while not launch._failing.locked():
                message = self.check()
                if message is not None:
                    self._expire(message)  # returns only for a test callback
                    return
                message = self._peers.check()
                if message is not None:
                    self._expire(message, launch.PeerFailedError)
                    return
                with self._lock:
                    deadline = self._deadline
                # Re-arm the GIL-independent backstop only when it would otherwise
                # fire less than backstop/2 after the current deadline. If none
                # can be armed at all, this thread still watches.
                if armed is None or deadline - armed > 0.5 * self.backstop_seconds:
                    launch._exit_later(max(deadline - self._clock(), 0.0) + self.backstop_seconds)
                    armed = deadline
                if self._stopped.wait(self.poll_seconds):
                    return
        finally:
            if not launch._failing.locked():  # keep fail_fast's own exit timer
                faulthandler.cancel_dump_traceback_later()

    def _expire(self, message, error=TimeoutError):
        if self._on_expire is not None:
            self._on_expire(message)
            return
        # Formatted by this thread (it holds the GIL) and written once with
        # rank-prefixed lines: faulthandler.dump_traceback needs a real stderr
        # descriptor, and its many small unprefixed writes interleave
        # unreadably when torchrun ranks share stderr.
        stacks = None
        try:
            stacks = _thread_stacks()
            sys.stdout.flush()
        except BaseException:
            pass  # e.g. sys.stdout is None; still report and exit below
        prefix = f"[rank {self.rank}] "
        launch._report("".join(prefix + line for line in (
            f"progress watchdog: {message}; stacks of all threads:\n",
            *(stacks or "").splitlines(True))))
        launch.fail_fast(error(f"progress watchdog: {message}"), rank=self.rank,
                         phase=self.phase_name, output=self.output, stacks=stacks)


def _thread_stacks() -> str:
    """Python stacks of every thread, most recent call last."""
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        parts.append(f"Thread {names.get(ident, '<unknown>')!r} (ident {ident}):\n")
        parts.extend(traceback.format_stack(frame))
    return "".join(parts)


def enter_long_phase(name: str):
    """Library hook: exempt a legitimately slow section (e.g. graph capture)."""
    watchdog = _active
    return None if watchdog is None else (watchdog, watchdog.enter(name))


def exit_long_phase(token) -> None:
    if token is not None:
        token[0].resume(token[1])


FAULT_ENV = "MATRIS_DIST_FAULT_INJECT"
# kind=skip faults that already fired (a skip is one-shot; the replay count it
# matches does not advance when the replay is skipped).
_SKIPPED = set()


@lru_cache(maxsize=4)
def _fault_spec(text):
    try:
        fields = dict(item.split("=", 1) for item in text.split(","))
        if set(fields) != {"rank", "phase", "step", "kind"} or fields["kind"] not in ("raise", "hang", "skip"):
            raise ValueError
        return int(fields["rank"]), fields["phase"], int(fields["step"]), fields["kind"]
    except ValueError:
        raise ValueError(f"{FAULT_ENV} must be 'rank=R,phase=P,step=S,kind=raise|hang|skip', "
                         f"got {text!r}") from None


def fault_point(phase: str, step: int, rank: int) -> bool:
    """TEST ONLY: fail one rank at a named site, e.g. ``rank=1,phase=evaluate,step=2,kind=hang``.

    A no-op (False) unless MATRIS_DIST_FAULT_INJECT is set. Sites: ``evaluate``
    (entry, step = evaluator epoch) and ``replay`` (before graph replay, step =
    replay count). ``kind=skip`` returns True: the ``replay`` site then skips
    that one replay (the silent-mismatch hazard of captured collectives).
    """
    text = os.environ.get(FAULT_ENV)
    if not text:
        return False
    target_rank, target_phase, target_step, kind = _fault_spec(text)
    if (target_rank, target_phase, target_step) != (rank, phase, step):
        return False
    sys.stderr.write(f"[rank {rank}] injecting test fault: {text}\n")
    sys.stderr.flush()
    if kind == "raise":
        raise RuntimeError(f"injected test fault ({FAULT_ENV}={text})")
    if kind == "skip":
        if text in _SKIPPED:
            return False
        _SKIPPED.add(text)
        return True
    while True:
        time.sleep(3600)
