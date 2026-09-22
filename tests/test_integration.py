"""Tests against the downloaded research data (skip in a source-only checkout)."""
import json
import pandas as pd
import pytest
from preprocessing import ROOT, load_dataset, temporal_split
from event_engine import ReplaySession, to_events
from predictor import Predictor

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "gradient_boosting.joblib").exists(),
    reason="Run download_data.py and train.py first")


def test_real_data_split_contains_no_future_training_labels():
    _, visits, examples, stats = load_dataset()
    train, test, cutoff, _ = temporal_split(examples)
    assert stats["raw_location_rows"] == 652
    assert stats["devices"] == 4
    assert train.target_time.max() < cutoff <= test.time.min()
    assert (examples.target_time > examples.time).all()
    assert (examples.next_cell != examples.current_cell).all()


def test_event_replay_matches_saved_holdout_predictions():
    _, visits, _, _ = load_dataset()
    predictor = Predictor()
    session = ReplaySession(to_events(visits[visits.time >= predictor.cutoff]), predictor,
                            to_events(visits[visits.time < predictor.cutoff]))
    session.subscribe()
    while session.step() is not None:
        pass
    actual = pd.DataFrame(session.analytics.resolved).sort_values(["time", "supi"])
    expected = pd.read_csv(ROOT / "results" / "test_predictions.csv", dtype=str)
    expected["time"] = pd.to_datetime(expected.time, format="mixed")
    expected = expected.sort_values(["time", "supi"])
    assert len(actual) == len(expected)
    assert actual.prediction.tolist() == expected["Gradient Boosting"].tolist()
    assert actual.actual.tolist() == expected.next_cell.tolist()
    report = json.loads((ROOT / "results" / "metrics.json").read_text())
    assert actual.correct.mean() == pytest.approx(report["models"]["Gradient Boosting"]["accuracy"])
