"""
Alert sinks: where scored alerts go.

Two real sinks ship, and the replay CLI uses both: StdoutSink prints one-line
alerts for terminal triage, JsonlSink appends one JSON object per line to a file
that a SIEM, `jq`, or `tail -f` can consume. That is the whole abstraction. There
is no plugin discovery, no queue, no database; a sink is anything with emit() and
close() (the AlertSink protocol), so a webhook or syslog sink is a small class
away when something real needs one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class AlertSink(Protocol):
    def emit(self, alert: dict) -> None: ...

    def close(self) -> None: ...


def format_alert(a: dict) -> str:
    mitre = f"{a['mitre_id']} {a['mitre_technique']}" if a["mitre_id"] else "n/a"
    esc = " [escalated->stage2]" if a["escalated"] else ""
    return (
        f"ALERT flow#{a['flow_index']:<5} {a['predicted_class']:<13} "
        f"conf={a['confidence']:.3f}{esc}  {mitre}  | {a['playbook']}"
    )


class StdoutSink:
    """One formatted line per alert, capped at max_alerts so a flood of 16k
    alerts stays readable; close() reports how many were suppressed."""

    def __init__(self, max_alerts: int = 25) -> None:
        self.max_alerts = max_alerts
        self._emitted = 0
        self._suppressed = 0

    def emit(self, alert: dict) -> None:
        if self._emitted < self.max_alerts:
            print(format_alert(alert))
            self._emitted += 1
        else:
            self._suppressed += 1

    def close(self) -> None:
        if self._suppressed:
            print(f"... {self._suppressed} more alerts not shown")


class JsonlSink:
    """One JSON object per line, appended. The structured twin of StdoutSink:
    everything the alert dict carries survives, nothing is truncated."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._count = 0

    def emit(self, alert: dict) -> None:
        self._fh.write(json.dumps(alert) + "\n")
        self._count += 1

    def close(self) -> None:
        self._fh.close()
        print(f"[sink] wrote {self._count} alerts to {self.path}")
