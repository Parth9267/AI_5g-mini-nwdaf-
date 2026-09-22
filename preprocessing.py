"""Past-only mobility features and a chronological, target-purged holdout."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CATEGORICAL = ["supi", "current_cell", "previous_cell", "time_period"]
NUMERIC = ["hour_sin", "hour_cos"]
FEATURES = CATEGORICAL + NUMERIC
PERIODS = ["night", "morning", "lunch", "afternoon", "evening"]


def feature_row(supi, current_cell, previous_cell, timestamp):
    """Only information available at the current event is used."""
    t = pd.Timestamp(timestamp)
    hour = t.hour + t.minute / 60
    period = ("night" if hour < 6 or hour >= 22 else "morning" if hour < 11
              else "lunch" if hour < 14 else "afternoon" if hour < 18 else "evening")
    return {"supi": str(supi), "current_cell": str(current_cell),
            "previous_cell": str(previous_cell), "time_period": period,
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24))}


def clean_locations(frame):
    required = {"time", "NrCellId", "supi", "tac"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Location data needs columns: {sorted(required)}")
    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="mixed").astype("datetime64[ns]")
    df = df.dropna(subset=list(required))
    for col in ["NrCellId", "supi", "tac"]:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col].ne("")]
    df["supi"] = df["supi"].str.removeprefix("imsi-")
    df = df.drop_duplicates(["supi", "time", "NrCellId"])
    # Conflicting simultaneous cell reports have no reliable temporal ordering.
    conflicts = df.groupby(["supi", "time"])["NrCellId"].transform("nunique") > 1
    return df.loc[~conflicts].sort_values(["supi", "time"]).reset_index(drop=True)


def build_visits(locations, registrations):
    """Segment at each observed inactive record; discard reports during inactivity.

    Registration timestamps are coarse. Before the first state report, accept
    locations as observed but do not infer a registration state. No use is made
    of duration_minutes, which could disclose future information.
    """
    reg = registrations.copy()
    reg["timestamp"] = pd.to_datetime(reg["timestamp"], format="mixed", errors="coerce").astype("datetime64[ns]")
    reg = reg.dropna(subset=["timestamp", "supi", "state_desc"])
    reg["supi"] = reg["supi"].astype(str).str.removeprefix("imsi-")
    reg["state_desc"] = reg["state_desc"].str.lower().str.strip()
    groups = []
    excluded = 0
    for device, group in locations.groupby("supi", sort=True):
        group = group.sort_values("time").copy()
        states = reg[reg.supi == device].sort_values("timestamp")
        inactive = states.loc[states.state_desc == "inactive", "timestamp"].to_numpy(dtype="datetime64[ns]")
        group["segment"] = np.searchsorted(inactive, group.time.to_numpy(dtype="datetime64[ns]"), side="right")
        if not states.empty:
            joined = pd.merge_asof(group, states[["timestamp", "state_desc"]],
                                   left_on="time", right_on="timestamp", direction="backward")
            keep = joined.state_desc.ne("inactive")
            excluded += int((~keep).sum())
            group = joined.loc[keep, ["time", "NrCellId", "supi", "tac", "segment"]].copy()
        changed = group.NrCellId.ne(group.NrCellId.shift()) | group.segment.ne(group.segment.shift())
        groups.append(group[changed])
    if not groups:
        raise ValueError("No valid location records were found.")
    visits = pd.concat(groups).sort_values(["time", "supi"]).reset_index(drop=True)
    visits = visits.rename(columns={"NrCellId": "current_cell"})
    return visits, excluded


def build_examples(visits):
    rows = []
    for (device, segment), group in visits.groupby(["supi", "segment"]):
        events = group.sort_values("time").to_dict("records")
        for i in range(1, len(events) - 1):
            previous, current, following = events[i - 1:i + 2]
            row = feature_row(device, current["current_cell"], previous["current_cell"], current["time"])
            row.update(time=current["time"], target_time=following["time"],
                       next_cell=following["current_cell"], segment=int(segment))
            rows.append(row)
    if not rows:
        raise ValueError("Not enough within-session movement for next-cell prediction.")
    return pd.DataFrame(rows).sort_values(["time", "supi"]).reset_index(drop=True)


def temporal_split(examples, train_fraction=0.7):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    ordered = examples.sort_values("time")
    cutoff = ordered.iloc[min(int(len(ordered) * train_fraction), len(ordered) - 1)].time
    # Purge examples whose future target has not been observed before cutoff.
    train = ordered[(ordered.time < cutoff) & (ordered.target_time < cutoff)].copy()
    test = ordered[ordered.time >= cutoff].copy()
    purged = len(ordered) - len(train) - len(test)
    if train.empty or test.empty:
        raise ValueError("This dataset cannot produce a nonempty chronological split.")
    return train, test, cutoff, purged


def load_dataset(raw_dir=None):
    folder = Path(raw_dir) if raw_dir else ROOT / "data" / "raw"
    locations = pd.read_csv(folder / "df_location.csv", dtype=str)
    registrations = pd.read_csv(folder / "df_reg.csv", dtype=str)
    clean = clean_locations(locations)
    visits, excluded = build_visits(clean, registrations)
    examples = build_examples(visits)
    stats = {"raw_location_rows": len(locations), "registration_rows": len(registrations),
             "clean_location_rows": len(clean), "inactive_reports_excluded": excluded,
             "visits": len(visits), "examples": len(examples),
             "devices": int(clean.supi.nunique()), "cells": int(clean.NrCellId.nunique())}
    return clean, visits, examples, stats
