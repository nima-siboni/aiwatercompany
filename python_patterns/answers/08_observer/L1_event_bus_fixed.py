"""
Reference fix for PR #281 (python_patterns/exercises/08_observer/L1_event_bus.py).
See the answer key for the reasoning.
"""
import logging
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Callable

logger = logging.getLogger(__name__)

Event = dict[str, Any]
Handler = Callable[[Event], None]

JOB_STATES = ("queued", "running", "finished", "failed")


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        self._subscribers[topic].remove(handler)

    def once(self, topic: str, handler: Handler) -> None:
        """Call `handler` for the next event on `topic` only."""
        def wrapper(event: Event) -> None:
            self.unsubscribe(topic, wrapper)
            handler(event)
        self.subscribe(topic, wrapper)

    def publish(self, topic: str, event: Event) -> None:
        # Iterate over a copy, because handlers may unsubscribe (see `once`) during delivery.
        for handler in list(self._subscribers.get(topic, ())):
            try:
                handler(event)
            except Exception:
                logger.exception("handler %r failed on %s", handler, topic)


class JobStateCounter:
    """Live number of jobs per state, for the dashboard."""

    def __init__(self, bus: EventBus):
        self.counts: Counter[str] = Counter()
        for state in JOB_STATES:
            bus.subscribe(f"job.{state}", partial(self._count, state))

    def _count(self, state: str, event: Event) -> None:
        self.counts[state] += 1


class ResultsWriter:
    """Stores the uptake of every finished job."""

    def __init__(self, bus: EventBus, store: dict[str, float]):
        self.bus, self.store = bus, store
        bus.subscribe("job.finished", self.on_finished)

    def on_finished(self, event: Event) -> None:
        self.store[event["job_id"]] = event["uptake"]

    def close(self) -> None:
        self.bus.unsubscribe("job.finished", self.on_finished)
