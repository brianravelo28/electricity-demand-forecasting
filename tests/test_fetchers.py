"""Tests for the data fetchers against mocked responses: no network, no API key, no downloaded data.

They cover paging, retries, what each fetcher does with a bad reply, and the unit conversions and
quality rules applied to weather observations.
"""
import time

import pandas as pd
import pytest
import requests

import eia
import ingest
import ingest_iem
import ingest_weather

KEY = "SECRETKEY123"


class Reply:
    def __init__(self, status=200, payload=None, text="", content=b""):
        self.status_code, self._payload, self.text, self.content = status, payload, text, content

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setenv("EIA_API_KEY", KEY)


# ---- EIA API helpers -------------------------------------------------------------------------

def test_get_sends_the_key_and_returns_the_response_part(monkeypatch):
    seen = {}

    def fake(url, params, timeout):
        seen["params"] = params
        return Reply(200, {"response": {"total": "1", "data": [{"value": "5"}]}})

    monkeypatch.setattr(requests, "get", fake)
    out = eia.get("https://api.example/data/", [("frequency", "hourly")])
    assert out["data"] == [{"value": "5"}]
    assert ("api_key", KEY) in seen["params"] and ("frequency", "hourly") in seen["params"]


def test_a_failed_request_never_shows_the_key(monkeypatch):
    def boom(url, params, timeout):
        raise requests.ConnectionError(f"could not reach https://x?api_key={KEY}")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(RuntimeError) as err:
        eia.get("https://api.example/data/", [])
    assert KEY not in str(err.value) and "***" in str(err.value)


def test_an_http_error_reports_the_status_and_hides_the_key(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: Reply(429, text=f"slow down api_key={KEY}"))
    with pytest.raises(RuntimeError) as err:
        eia.get("https://api.example/data/", [])
    assert "HTTP 429" in str(err.value) and KEY not in str(err.value)


def test_demand_params_ask_for_one_ba_and_type_sorted_by_period():
    p = dict(eia.demand_params("ERCO", "D", start="2019-01-01T00", offset=5000))
    assert p["facets[respondent][]"] == "ERCO" and p["facets[type][]"] == "D"
    assert p["sort[0][column]"] == "period" and p["offset"] == 5000 and p["start"] == "2019-01-01T00"
    assert "end" not in p


def test_fetch_all_reads_every_page_and_stops_at_the_total(monkeypatch):
    offsets = []

    def fake_get(route, params):
        offset = dict(params)["offset"]
        offsets.append(offset)
        n = min(eia.PAGE, 12000 - offset)
        return {"total": "12000", "data": [{"period": "p", "value": str(offset + i)} for i in range(n)]}

    monkeypatch.setattr(eia, "get", fake_get)
    df = eia.fetch_all("ERCO", "D", "2019-01-01T00", None)
    assert offsets == [0, 5000, 10000] and len(df) == 12000
    assert df["value"].iloc[-1] == "11999"


def test_a_200_reply_without_a_response_part_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: Reply(200, {"error": "no such route"}))
    with pytest.raises(RuntimeError, match="no 'response' part"):
        eia.get("https://api.example/data/", [])


def test_a_200_reply_that_is_not_json_raises_a_clear_error_and_hides_the_key(monkeypatch):
    class NotJson(Reply):
        def json(self):
            raise ValueError("Expecting value")

    monkeypatch.setattr(requests, "get", lambda url, params, timeout: NotJson(200, text=f"<html>maintenance {KEY}</html>"))
    with pytest.raises(RuntimeError, match="not JSON") as err:
        eia.get("https://api.example/data/", [])
    assert KEY not in str(err.value)


def test_a_reply_body_that_echoes_the_key_is_scrubbed(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: Reply(200, {"note": f"api_key={KEY}"}))
    with pytest.raises(RuntimeError) as err:
        eia.get("https://api.example/data/", [])
    assert KEY not in str(err.value)


def test_fetch_all_fails_when_fewer_rows_arrive_than_the_api_reports(monkeypatch):
    def short(route, params):
        offset = dict(params)["offset"]
        n = 2000 if offset == 5000 else eia.PAGE  # the second page should hold 3,000 rows and holds 2,000
        return {"total": "8000", "data": [{"value": str(i)} for i in range(n)]}

    monkeypatch.setattr(eia, "get", short)
    with pytest.raises(RuntimeError, match=r"Incomplete download for ERCO D: 7000 rows received, the API reports 8000"):
        eia.fetch_all("ERCO", "D", "2019-01-01T00", None)


def test_fetch_all_reports_the_counts_in_an_incomplete_download(monkeypatch):
    def short(route, params):
        return {"total": "100", "data": [{"value": "1"}] * 60}

    monkeypatch.setattr(eia, "get", short)
    with pytest.raises(RuntimeError, match=r"60 rows received, the API reports 100"):
        eia.fetch_all("ERCO", "D", "2019-01-01T00", None)


def test_fetch_all_accepts_a_total_that_grows_while_paging(monkeypatch):
    """A new hour can arrive between pages; the last reply's total is the one to match."""
    def growing(route, params):
        if dict(params)["offset"] == 0:
            return {"total": "5001", "data": [{"value": "a"}] * eia.PAGE}
        return {"total": "5002", "data": [{"value": "b"}] * 2}

    monkeypatch.setattr(eia, "get", growing)
    assert len(eia.fetch_all("ERCO", "D", "2019-01-01T00", None)) == 5002


def test_fetch_all_rejects_a_reply_without_data_or_total(monkeypatch):
    monkeypatch.setattr(eia, "get", lambda route, params: {"warnings": ["slow"]})
    with pytest.raises(RuntimeError, match="no data or total"):
        eia.fetch_all("ERCO", "D", "2019-01-01T00", None)


def test_fetch_all_with_no_rows_returns_an_empty_frame(monkeypatch):
    monkeypatch.setattr(eia, "get", lambda route, params: {"total": "0", "data": []})
    assert len(eia.fetch_all("ERCO", "D", "2019-01-01T00", None)) == 0


# ---- ingest.py -------------------------------------------------------------------------------

def test_the_retry_wrapper_retries_rate_limits_and_server_errors(monkeypatch):
    calls = []

    def flaky(code, kind, start, end):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("HTTP 429 from x: slow down")
        if len(calls) == 2:
            raise RuntimeError("HTTP 503 from x: busy")
        return pd.DataFrame({"value": ["1"]})

    monkeypatch.setattr(ingest, "fetch_all", flaky)
    assert len(ingest.fetch_with_retry("ERCO", "D")) == 1 and len(calls) == 3


def test_the_retry_wrapper_does_not_retry_other_errors(monkeypatch):
    calls = []

    def bad(code, kind, start, end):
        calls.append(1)
        raise RuntimeError("HTTP 400 from x: bad facet")

    monkeypatch.setattr(ingest, "fetch_all", bad)
    with pytest.raises(RuntimeError, match="400"):
        ingest.fetch_with_retry("ERCO", "D")
    assert len(calls) == 1


def test_the_retry_wrapper_gives_up_after_four_tries(monkeypatch):
    calls = []

    def down(code, kind, start, end):
        calls.append(1)
        raise RuntimeError("HTTP 503 from x: busy")

    monkeypatch.setattr(ingest, "fetch_all", down)
    with pytest.raises(RuntimeError, match="503"):
        ingest.fetch_with_retry("ERCO", "D")
    assert len(calls) == 4


def test_save_writes_utc_periods_numeric_values_and_leaves_no_temp_file(tmp_path):
    df = pd.DataFrame({"respondent": ["ERCO"] * 3, "type": ["D"] * 3,
                       "period": ["2024-03-10T06", "2024-03-10T07", "2024-03-10T08"],
                       "value": ["100", None, "not a number"]})
    path = tmp_path / "ERCO_D.parquet"
    assert ingest.save(df, path) == 3
    out = pd.read_parquet(path)
    assert str(out["period"].dt.tz) == "UTC" and out["period"].iloc[1] == pd.Timestamp("2024-03-10 07:00", tz="UTC")
    assert out["value"].iloc[0] == 100 and out["value"].iloc[1:].isna().all()
    assert not path.with_suffix(".tmp").exists()


# ---- Iowa Mesonet observations ---------------------------------------------------------------

def mesonet(rows):
    return "station,valid,tmpf,dwpf,sknt\n" + "\n".join(f"KXXX,{v},{t},{d},{s}" for v, t, d, s in rows)


def test_mesonet_values_are_converted_to_celsius_and_metres_per_second():
    out, flagged = ingest_iem.parse(mesonet([("2025-09-01 00:00", "50", "41", "10")]))
    assert out["temp_c"].iloc[0] == 10.0 and out["dew_c"].iloc[0] == 5.0
    assert out["wind_ms"].iloc[0] == pytest.approx(5.1, abs=0.05) and flagged == 0
    assert str(out["time"].dt.tz) == "UTC"


def test_mesonet_missing_markers_become_missing_values_and_are_not_flagged():
    out, flagged = ingest_iem.parse(mesonet([("2025-09-01 00:00", "M", "M", "M")]))
    assert out[["temp_c", "dew_c", "wind_ms"]].isna().all(axis=None) and flagged == 0


def test_mesonet_impossible_readings_are_set_to_missing_and_counted():
    rows = [("2025-09-01 00:00", "200", "41", "10"),   # 93 C
            ("2025-09-01 01:00", "50", "60", "10"),    # dew point far above the temperature
            ("2025-09-01 02:00", "50", "41", "200"),   # 103 m/s
            ("2025-09-01 03:00", "50", "41", "10")]    # fine
    out, flagged = ingest_iem.parse(mesonet(rows))
    assert flagged == 3
    assert pd.isna(out["temp_c"].iloc[0]) and pd.isna(out["dew_c"].iloc[1]) and pd.isna(out["wind_ms"].iloc[2])
    assert out.loc[3, ["temp_c", "dew_c", "wind_ms"]].notna().all()


def test_mesonet_fetch_retries_a_server_error_then_succeeds(monkeypatch):
    replies = iter([Reply(500), Reply(200, text="csv text")])
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: next(replies))
    assert ingest_iem.fetch("KXXX", pd.Timestamp("2025-09-02").date()) == "csv text"


def test_mesonet_fetch_raises_when_the_server_keeps_failing(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: Reply(500))
    with pytest.raises(requests.HTTPError):
        ingest_iem.fetch("KXXX", pd.Timestamp("2025-09-02").date())


def test_mesonet_fetch_raises_after_repeated_connection_failures(monkeypatch):
    def down(url, params, timeout):
        raise requests.ConnectionError("reset")

    monkeypatch.setattr(requests, "get", down)
    with pytest.raises(requests.ConnectionError):
        ingest_iem.fetch("KXXX", pd.Timestamp("2025-09-02").date())


# ---- NOAA ISD observations -------------------------------------------------------------------

def test_isd_fields_drop_suspect_values_and_the_missing_sentinel_but_keep_unchecked_ones():
    s = pd.Series(["+0123,1", "+9999,9", "+0200,2", "+0300,9", "-0050,7"])
    value, quality = ingest_weather.parse_field(s, 10, 9999)
    assert value.iloc[0] == pytest.approx(12.3)
    assert pd.isna(value.iloc[1])                  # the missing sentinel
    assert pd.isna(value.iloc[2]) and pd.isna(value.iloc[4])  # quality codes 2 and 7: suspect or erroneous
    assert value.iloc[3] == pytest.approx(30.0)    # quality 9: passed gross limits, kept
    assert list(quality) == ["1", "9", "2", "9", "7"]


def isd_csv(rows):
    header = "DATE,REPORT_TYPE,TMP,DEW,WND"
    return (header + "\n" + "\n".join(f'{d},{t},"{tmp}","{dew}","{wnd}"' for d, t, tmp, dew, wnd in rows)).encode()


def test_isd_rows_are_parsed_to_celsius_and_metres_per_second():
    raw = isd_csv([("2024-01-01T01:00:00", "FM-15", "+0123,1", "+0050,1", "180,1,N,0050,1"),
                   ("2024-01-01T00:00:00", "FM-15", "+0100,1", "+9999,9", "999,9,9,9999,9")])
    out = ingest_weather.parse(raw)
    assert list(out["time"].dt.hour) == [0, 1]  # sorted by time
    assert out["temp_c"].tolist() == pytest.approx([10.0, 12.3])
    assert pd.isna(out["dew_c"].iloc[0]) and out["dew_c"].iloc[1] == pytest.approx(5.0)
    assert pd.isna(out["wind_ms"].iloc[0]) and out["wind_ms"].iloc[1] == pytest.approx(5.0)


def test_isd_fetch_returns_none_for_a_missing_file_and_the_bytes_otherwise(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout: Reply(404))
    assert ingest_weather.fetch("https://x/2019/A.csv") is None
    monkeypatch.setattr(requests, "get", lambda url, timeout: Reply(200, content=b"abc"))
    assert ingest_weather.fetch("https://x/2019/A.csv") == b"abc"


def test_isd_fetch_retries_a_server_error_then_raises_if_it_persists(monkeypatch):
    replies = iter([Reply(503), Reply(200, content=b"ok")])
    monkeypatch.setattr(requests, "get", lambda url, timeout: next(replies))
    assert ingest_weather.fetch("https://x/2019/A.csv") == b"ok"
    monkeypatch.setattr(requests, "get", lambda url, timeout: Reply(503))
    with pytest.raises(requests.HTTPError):
        ingest_weather.fetch("https://x/2019/A.csv")
