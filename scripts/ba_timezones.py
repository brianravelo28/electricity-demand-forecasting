"""Civil time zone of each selected BA's load, for calendar features (hour, weekday, holidays).

This is the clock people in the load area follow, not necessarily the zone EIA lists. EIA data
itself is on UTC, so these zones only decide the local hour and date. Evidence is in
scripts/timezone_check.py: the winter morning ramp and the shift around the March clock change.
"""

BA_TIMEZONE = {
    "PJM": "America/New_York",
    "MISO": "America/Chicago",     # EIA lists Eastern and reports EST all year; the load follows Central
    "ERCO": "America/Chicago",
    "SWPP": "America/Chicago",
    "SOCO": "America/New_York",    # EIA lists Central; the morning ramp matches the Eastern BAs
    "CISO": "America/Los_Angeles",
    "TVA": "America/Chicago",      # spans Central and Eastern; evidence is mixed, EIA lists Central
    "NYIS": "America/New_York",
    "FPL": "America/New_York",
    "ISNE": "America/New_York",
    "DUK": "America/New_York",
    "CPLE": "America/New_York",
    "BPAT": "America/Los_Angeles",
    "FPC": "America/New_York",
    "PSCO": "America/Denver",
    "PACE": "America/Denver",
    "NEVP": "America/Los_Angeles",
    "LGEE": "America/New_York",    # EIA reports EST all year; the load follows daylight saving
    "SRP": "America/Phoenix",      # Arizona does not observe daylight saving
    "AZPS": "America/Phoenix",
}
