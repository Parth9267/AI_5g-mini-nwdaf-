"""Descriptive statistics for observed within-session cell transitions."""
import pandas as pd


def transitions(visits):
    df = visits.sort_values(["supi", "segment", "time"]).copy()
    grouped = df.groupby(["supi", "segment"])
    df["from_cell"] = grouped.current_cell.shift()
    df["previous_time"] = grouped.time.shift()
    df = df.dropna(subset=["from_cell"])
    df["gap_minutes"] = (df.time - df.previous_time).dt.total_seconds() / 60
    df["route"] = df.from_cell.str.lstrip("0") + " → " + df.current_cell.str.lstrip("0")
    return df
