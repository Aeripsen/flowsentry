"""
Map FlowSentry's five output classes to a representative MITRE ATT&CK technique
plus a one-line response playbook.

HONESTY: this mapping is CLASS-level, not per-signature. FlowSentry predicts one of
five coarse NSL-KDD families (dos, probe, r2l, u2r, normal), so each family points
at ONE representative technique that fits the family as a whole. A real SOC would
refine this per signature (for example neptune and smurf are both "dos" but a flood
and a reflection attack respond differently). Treat the technique id as a triage
hint, not a forensic conclusion.

Technique choices (T-ids from attack.mitre.org):
  dos   -> T1498 Network Denial of Service
  probe -> T1046 Network Service Discovery
  r2l   -> T1110 Brute Force            (r2l also overlaps T1078 Valid Accounts when
                                         a guessed credential succeeds)
  u2r   -> T1068 Exploitation for Privilege Escalation
  normal-> none
"""
from __future__ import annotations

# class -> {representative MITRE technique id/name, one-line response playbook}.
ATTACK_MAP: dict[str, dict[str, str | None]] = {
    "dos": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "playbook": (
            "Rate-limit or blackhole the source, page on-call netops, "
            "confirm upstream scrubbing is on."
        ),
    },
    "probe": {
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "playbook": (
            "Log and watch the scanning source, tighten exposed-service firewall "
            "rules, alert if a scan is followed by a connect."
        ),
    },
    "r2l": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "playbook": (
            "Force a reset on the targeted account, enable lockout and MFA, "
            "review auth logs for a successful login."
        ),
    },
    "u2r": {
        "technique_id": "T1068",
        "technique_name": "Exploitation for Privilege Escalation",
        "playbook": (
            "Isolate the host, snapshot it for forensics, hunt for new root "
            "processes and persistence, rotate host secrets."
        ),
    },
    "normal": {
        "technique_id": None,
        "technique_name": None,
        "playbook": "No action; benign traffic.",
    },
}

# UNKNOWN/abstained predictions and any label outside the five families land here.
_UNMAPPED: dict[str, str | None] = {
    "technique_id": None,
    "technique_name": None,
    "playbook": "Model abstained or label is unmapped; route to a human analyst for review.",
}


def lookup(label: str) -> dict[str, str | None]:
    """Return the MITRE technique + playbook for a predicted class label.

    Unknown/abstained or any label outside the five families falls back to a
    human-review entry rather than raising, so the stream never crashes on a
    surprise label.
    """
    return ATTACK_MAP.get(label, _UNMAPPED)
