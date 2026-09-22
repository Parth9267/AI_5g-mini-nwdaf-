"""Small in-process publish/subscribe replay of recorded location visits.

No HTTP, AMF, radio or 3GPP compliance is implied. Publishers never supply
the next-cell label. A forecast is scored only when a later event arrives.
"""
from uuid import uuid4


class EventBus:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type, callback):
        token = str(uuid4())
        self.subscribers[token] = (event_type, callback)
        return token

    def unsubscribe(self, token):
        return self.subscribers.pop(token, None) is not None

    def publish(self, event):
        delivered = 0
        for event_type, callback in list(self.subscribers.values()):
            if event["type"] == event_type:
                callback(dict(event))
                delivered += 1
        return delivered


class MobilityAnalytics:
    def __init__(self, predictor):
        self.predictor = predictor
        self.history = {}
        self.pending = {}
        self.resolved = []
        self.received = 0
        self.latest = None

    def seed(self, events):
        """Warm history using only observations before evaluation cutoff."""
        for event in events:
            self.history[event["supi"]] = event

    def clear_context(self):
        # Unsubscribed events are missed, so continuity can no longer be assumed.
        self.history.clear()
        self.pending.clear()
        self.latest = None

    def on_event(self, event):
        self.received += 1
        device = event["supi"]
        previous = self.history.get(device)
        pending = self.pending.pop(device, None)
        continuous = previous is not None and previous["segment"] == event["segment"]
        if continuous and previous["cell"] == event["cell"]:
            if pending:
                self.pending[device] = pending
            return
        if pending and continuous:
            self.resolved.append({**pending, "actual": event["cell"],
                                  "resolved_at": event["time"],
                                  "correct": pending["prediction"] == event["cell"]})
        result = None
        if continuous:
            result = self.predictor.predict(device, event["cell"], previous["cell"], event["time"])
            self.pending[device] = {"supi": device, "time": event["time"], "from_cell": event["cell"],
                                    "prediction": result["cell"], "score": result["score"]}
        self.history[device] = event
        self.latest = {**event, "previous_cell": previous["cell"] if continuous else None, "forecast": result}


def to_events(visits):
    return [{"type": "LOCATION_REPORT", "time": row.time, "supi": row.supi,
             "cell": row.current_cell, "segment": int(row.segment)}
            for row in visits.itertuples()]


class ReplaySession:
    def __init__(self, events, predictor, seed_events=()):
        self.events = list(events)
        self.cursor = 0
        self.bus = EventBus()
        self.analytics = MobilityAnalytics(predictor)
        self.analytics.seed(seed_events)
        self.subscription_id = None
        self.log = []

    def subscribe(self):
        if self.subscription_id is None:
            self.subscription_id = self.bus.subscribe("LOCATION_REPORT", self.analytics.on_event)
        return self.subscription_id

    def unsubscribe(self):
        if self.subscription_id is not None:
            self.bus.unsubscribe(self.subscription_id)
            self.subscription_id = None
            self.analytics.clear_context()

    def step(self):
        if self.cursor >= len(self.events):
            return None
        event = self.events[self.cursor]
        self.cursor += 1
        delivered = self.bus.publish(event)
        if not delivered:
            self.analytics.clear_context()
        self.log.append({**event, "delivered": bool(delivered)})
        return event
