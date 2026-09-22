import pandas as pd
from preprocessing import clean_locations, build_visits, build_examples, temporal_split, feature_row


def locations(rows):
    return clean_locations(pd.DataFrame(rows, columns=["time", "NrCellId", "supi", "tac"]))


def test_session_boundaries_and_inactive_reports_do_not_make_handovers():
    loc = locations([(f"2025-01-01 {t}", c, "001", "01") for t, c in
                     [("08:00", "30"), ("08:05", "30"), ("08:10", "40"),
                      ("08:20", "50"), ("09:05", "60"), ("09:10", "30"), ("09:20", "40")]])
    reg = pd.DataFrame({"timestamp": ["2025-01-01 08:15", "2025-01-01 09:00"],
                        "supi": ["imsi-001"] * 2, "state_desc": ["inactive", "active"]})
    visits, excluded = build_visits(loc, reg)
    assert excluded == 1
    assert visits.current_cell.tolist() == ["30", "40", "60", "30", "40"]
    examples = build_examples(visits)
    assert len(examples) == 1
    assert examples.iloc[0][["previous_cell", "current_cell", "next_cell"]].tolist() == ["60", "30", "40"]


def test_cleaning_preserves_ids_and_drops_ambiguous_simultaneous_reports():
    frame = locations([("2025-01-01 08:00", "030", "0001", "001"),
                       ("2025-01-01 08:00", "030", "0001", "001"),
                       ("2025-01-01 08:05", "040", "0001", "001"),
                       ("2025-01-01 08:05", "050", "0001", "001"),
                       ("bad-date", "060", "0001", "001")])
    assert len(frame) == 1
    assert frame.iloc[0].supi == "0001"
    assert frame.iloc[0].NrCellId == "030"


def test_split_purges_future_labels():
    times = pd.date_range("2025-01-01", periods=10, freq="h")
    data = pd.DataFrame({"time": times, "target_time": times + pd.Timedelta(hours=2)})
    train, test, cutoff, purged = temporal_split(data)
    assert (train.target_time < cutoff).all()
    assert (test.time >= cutoff).all()
    assert purged == 2


def test_features_cannot_access_future_target():
    row = feature_row("001", "40", "30", "2025-01-01 12:00")
    assert row["time_period"] == "lunch"
    assert "next_cell" not in row and "duration_minutes" not in row
