"""Demand outlier cleaning (see docs/PLANNING.md).

EIA's hourly demand has impossible readings: values near 2^31, negative values, sudden
spikes and drop-outs that last an hour or a few. Raw files stay untouched; the cleaned
series is the author's, with the removed readings set to missing.

The rule looks backward only, so cleaning a reading never uses anything after it:
a reading is removed when it is not positive, or when it is below LOW times or above
HIGH times the median of the previous WINDOW hours (readings already removed are still
in that median, which is robust to a minority of bad hours).
"""
import numpy as np
import pandas as pd

WINDOW = 15 * 24  # hours of history for the reference median
MIN_HISTORY = 48  # no judgement until this many earlier readings exist
LOW = 0.4
HIGH = 2.0


def clean_demand(demand):
    """Return (cleaned series, boolean Series of removed readings) on the same index."""
    demand = demand.astype(float)
    ref = demand.where(demand > 0).rolling(WINDOW, min_periods=MIN_HISTORY).median().shift(1)
    ratio = demand / ref
    bad = ~(demand > 0) & demand.notna()
    bad |= (ratio < LOW) | (ratio > HIGH)
    return demand.mask(bad), bad
