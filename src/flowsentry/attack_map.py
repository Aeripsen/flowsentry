"""
Map FlowSentry's output classes to a representative MITRE ATT&CK technique plus a
one-line response playbook.

HONESTY: this mapping is CLASS-level, not per-signature. FlowSentry predicts benign
or one of seven named UDP DDoS families. Every attack family in this dataset is a
volumetric UDP flood, so they all map to the same top-level technique, T1498 Network
Denial of Service (sub-technique T1498.001 Direct Network Flood). The family name is
the specific campaign/tool (VSE, OVH, HULK, RAW, ...); the ATT&CK id is the behaviour.
Treat the technique id as a triage hint, not a forensic conclusion.

Technique choices (T-ids from attack.mitre.org):
  UDP-* flood families -> T1498.001 Direct Network Flood (under T1498 Network DoS)
  benign               -> none
"""
from __future__ import annotations

_UDP_FLOOD = {
    "technique_id": "T1498.001",
    "technique_name": "Network Denial of Service: Direct Network Flood",
    "playbook": (
        "Rate-limit or blackhole the source, engage upstream/cloud DDoS scrubbing, "
        "page on-call netops, and confirm the target service is still reachable."
    ),
}

# Each named UDP campaign is a volumetric flood -> the same ATT&CK behaviour.
ATTACK_MAP: dict[str, dict[str, str | None]] = {
    fam: dict(_UDP_FLOOD)
    for fam in (
        "UDP-RAW", "UDP-VSE", "UDP-OVH", "UDP-MULTI", "UDP-HULK", "UDP-bypass-v1",
        "UDP-GAME",
    )
}
ATTACK_MAP["benign"] = {
    "technique_id": None,
    "technique_name": None,
    "playbook": "No action; benign traffic.",
}

# UNKNOWN/abstained predictions and any label outside the closed set land here.
_UNMAPPED: dict[str, str | None] = {
    "technique_id": None,
    "technique_name": None,
    "playbook": "Model abstained or label is unmapped; route to a human analyst for review.",
}


def lookup(label: str) -> dict[str, str | None]:
    """Return the MITRE technique + playbook for a predicted class label.

    Unknown/abstained or any out-of-set label falls back to a human-review entry
    rather than raising, so the stream never crashes on a surprise label.
    """
    return ATTACK_MAP.get(label, _UNMAPPED)
