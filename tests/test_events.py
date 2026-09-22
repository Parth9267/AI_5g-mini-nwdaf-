from event_engine import EventBus, ReplaySession


class FakePredictor:
    def predict(self, *args):
        return {"cell": "50", "score": .7, "probabilities": {"50": .7, "30": .3}}


def event(cell, segment=0):
    return {"type": "LOCATION_REPORT", "supi": "001", "cell": cell, "segment": segment, "time": "2025-01-01 12:00"}


def test_bus_filters_events_and_unsubscribe_stops_delivery():
    bus = EventBus()
    received = []
    token = bus.subscribe("LOCATION_REPORT", received.append)
    assert bus.publish({"type": "OTHER"}) == 0
    assert bus.publish(event("30")) == 1
    bus.unsubscribe(token)
    assert bus.publish(event("40")) == 0
    assert len(received) == 1


def test_prediction_is_resolved_only_after_next_observation():
    replay = ReplaySession([event("40"), event("50")], FakePredictor(), [event("30")])
    assert replay.subscribe() == replay.subscribe()
    replay.step()
    assert len(replay.analytics.resolved) == 0
    assert replay.analytics.pending["001"]["prediction"] == "50"
    replay.step()
    assert replay.analytics.resolved[0]["correct"] is True
    assert replay.step() is None


def test_new_session_does_not_score_old_pending_prediction():
    replay = ReplaySession([event("40"), event("50", 1)], FakePredictor(), [event("30")])
    replay.subscribe()
    replay.step()
    replay.step()
    assert replay.analytics.resolved == []
    assert replay.analytics.latest["forecast"] is None


def test_unsubscribed_events_do_not_update_analytics_and_break_history():
    replay = ReplaySession([event("40"), event("50"), event("60")], FakePredictor(), [event("30")])
    replay.subscribe()
    replay.step()
    replay.unsubscribe()
    replay.step()
    assert replay.analytics.received == 1
    replay.subscribe()
    replay.step()
    assert replay.analytics.received == 2
    assert replay.analytics.resolved == []
    assert replay.analytics.latest["forecast"] is None
