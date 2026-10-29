"""
PR #281: In-process event bus for simulation-job events

When a simulation job changes state, several services need to know: the
results writer stores the computed uptake, the dashboard counts jobs per
state, and the active-learning loop retrains once enough new labels arrive.
Instead of the job runner calling each of them directly, it now publishes
events such as `bus.publish("job.finished", {"job_id": "j42", "uptake": 2.1})`,
and each service subscribes to the topics it needs.
"""
from collections import Counter, defaultdict
from typing import Any, Callable

Event = dict[str, Any]
Handler = Callable[[Event], None]

JOB_STATES = ("queued", "running", "finished", "failed")


class EventBus:
    _subscribers: dict[str, list[Handler]] = defaultdict(list)

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
        for handler in self._subscribers[topic]:
            handler(event)


class JobStateCounter:
    """Live number of jobs per state, for the dashboard."""

    def __init__(self, bus: EventBus):
        self.counts: Counter[str] = Counter()
        for state in JOB_STATES:
            def on_event(event: Event) -> None:
                self.counts[state] += 1
            bus.subscribe(f"job.{state}", on_event)


class ResultsWriter:
    """Stores the uptake of every finished job."""

    def __init__(self, bus: EventBus, store: dict[str, float]):
        self.bus, self.store = bus, store
        bus.subscribe("job.finished", self.on_finished)

    def on_finished(self, event: Event) -> None:
        self.store[event["job_id"]] = event["uptake"]

    def close(self) -> None:
        self.bus.unsubscribe("job.finished", self.on_finished)
