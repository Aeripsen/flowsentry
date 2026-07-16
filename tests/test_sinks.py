"""Sink tests: both shipped sinks satisfy the protocol and do their one job."""
import json

from flowsentry.sinks import AlertSink, JsonlSink, StdoutSink


def _alert(i: int) -> dict:
    return {
        "timestamp": "2026-07-16T00:00:00.000+00:00",
        "flow_index": i,
        "predicted_class": "UDP-RAW",
        "confidence": 0.99,
        "escalated": False,
        "abstained": False,
        "mitre_id": "T1498.001",
        "mitre_technique": "Network Denial of Service: Direct Network Flood",
        "playbook": "Rate-limit or blackhole the source.",
        "true_label": "UDP-RAW",
    }


def test_both_sinks_satisfy_the_protocol():
    assert isinstance(StdoutSink(), AlertSink)


def test_jsonl_sink_satisfies_protocol_and_roundtrips(tmp_path):
    path = tmp_path / "alerts.jsonl"
    sink = JsonlSink(path)
    assert isinstance(sink, AlertSink)
    alerts = [_alert(i) for i in range(3)]
    for a in alerts:
        sink.emit(a)
    sink.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    # the [sink] summary goes to stdout, not into the file
    assert len(lines) == 3
    assert [json.loads(line) for line in lines] == alerts


def test_jsonl_sink_appends_not_truncates(tmp_path):
    path = tmp_path / "alerts.jsonl"
    for round_ in range(2):
        sink = JsonlSink(path)
        sink.emit(_alert(round_))
        sink.close()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_stdout_sink_caps_output_and_reports_suppressed(capsys):
    sink = StdoutSink(max_alerts=2)
    for i in range(5):
        sink.emit(_alert(i))
    sink.close()
    out = capsys.readouterr().out
    assert out.count("ALERT") == 2
    assert "3 more alerts not shown" in out


def test_stdout_sink_quiet_when_nothing_suppressed(capsys):
    sink = StdoutSink(max_alerts=10)
    sink.emit(_alert(0))
    sink.close()
    out = capsys.readouterr().out
    assert out.count("ALERT") == 1
    assert "not shown" not in out
