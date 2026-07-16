"""
BCCC-UDP-QUIC-IDS-2025 loader + the protocol-aware feature schema.

This is the real dataset from the SECRYPT 2026 paper
"Unveiling Hierarchical Machine Learning UDP-QUIC Intrusion Detection"
(Jafari, Shafi, Habibi Lashkari). Flows were extracted from cloud PCAP captures
by Sepehr's own analyzers: UDPFlowLyzer (UDP flow statistics) and QUICFlowLyzer
(QUIC-specific metadata). The dataset is released under CC BY 4.0.

The repo ships a stratified subset of the public dataset
(`data/sample/bccc_udp_quic_sample.csv.gz`) so the whole pipeline reproduces from
a clean clone without the multi-GB full download. The subset keeps ALL flows of
the rare attack families and caps the two dominant classes (benign, UDP-RAW), so
metrics are computed on a balanced, non-degenerate slice. See docs/MODEL_CARD.md
for how the full-dataset numbers compare.

Two feature sets drive the two-stage model, defined BY NAME below:

  STAGE1_FEATURES = the 114 UDP flow statistics from UDPFlowLyzer. These are the
      cheap, always-available features present on every UDP flow (packet/byte
      counts, rates, inter-arrival-time stats, burst and idle structure, size
      distribution moments and percentiles, entropy, directional asymmetry).
      Stage 1 (fast UDP-only model) runs on exactly these.

  STAGE2_FEATURES = STAGE1_FEATURES + the 18 QUIC features from QUICFlowLyzer
      (has_quic_subflow, quic_* handshake/timing/path-migration signals). Stage 2
      (the QUIC-augmented fallback) runs on the joint set, and is only invoked for
      flows Stage 1 could not classify confidently.

Identifier and label columns (flow_id, timestamp, IPs, ports, protocol,
attack_type) are intentionally EXCLUDED from the feature space to avoid trivial
shortcuts and label leakage. Ports/IPs are kept only to form the connection key
for the leakage-safe split, never as model inputs.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SAMPLE_PATH = DATA_DIR / "sample" / "bccc_udp_quic_sample.csv.gz"

# Columns present in the CSV that are NOT model features.
ID_COLS = ["flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
LABEL_COLS = ["label", "attack_type", "label_mc"]
# The 4-tuple used only to build the connection group key for the leakage-safe split.
CONN_KEYS = ["src_ip", "src_port", "dst_ip", "dst_port"]
TARGET = "label_mc"

# Closed-set attack families kept for the multi-class task: benign + the seven named
# UDP DDoS campaigns present in the sample. The dataset's coarse catch-all labels
# ("attack", "suspicious") are excluded from the closed-set family task exactly as in
# the paper's attack-family table; they are ambiguous by construction (attack-day
# background that could not be matched to a specific campaign window).
FAMILIES = [
    "benign",
    "UDP-RAW",
    "UDP-VSE",
    "UDP-OVH",
    "UDP-MULTI",
    "UDP-HULK",
    "UDP-bypass-v1",
    "UDP-GAME",
]

# The 18 QUIC features contributed by QUICFlowLyzer (the Stage-2 augmentation).
QUIC_FEATURES = [
    "has_quic_subflow", "quic_match_count", "quic_payload_bytes_sum", "quic_packets_sum",
    "quic_duration_sum", "quic_bytes_rate_mean", "quic_packets_rate_mean",
    "quic_iat_mean_mean", "quic_iat_std_mean", "quic_iat_p50_mean", "quic_iat_p90_mean",
    "quic_used0rtt_any", "quic_had_retry_any", "quic_paths_max", "quic_migrations_sum",
    "quic_dcid_rotations_client_sum", "quic_dcid_rotations_server_sum", "quic_min_time_diff",
]

# The 114 UDP flow statistics from UDPFlowLyzer (the Stage-1 feature set). Listed by
# name so the model can be defended feature by feature. Every one is a numeric
# statistic computed per UDP flow; none is a one-hot dummy.
UDP_FEATURES = [
    "duration", "pkt_count", "byte_count", "pps", "bps", "avg_pkt_size", "size_var",
    "min_pkt", "max_pkt", "pkt_size_skew", "pkt_size_kurt", "total_iat", "mean_iat",
    "iat_std", "iat_skew", "iat_kurt", "jitter", "burst_cnt", "mean_burst_len",
    "idle_ratio", "hurst_exponent", "median_iat", "frag_ratio", "ipv4_frag_ratio",
    "fragments_per_flow", "dest_multicast_flag", "multicast_ratio",
    "udp_length_mismatch_ratio", "dscp_diversity", "tos_mode", "ipid_increment_var",
    "pkt_count_fwd", "pkt_count_bwd", "byte_count_fwd", "byte_count_bwd",
    "payload_bytes_fwd", "payload_bytes_bwd", "fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio",
    "payload_efficiency", "header_overhead_ratio", "directional_asymmetry",
    "median_pkt_size", "pkt_size_cov", "payload_size_skew", "payload_size_kurt",
    "header_size_skew", "header_size_kurt", "size_time_correlation", "pkt_size_var",
    "pkt_size_range", "delta_pkt_size_min", "delta_pkt_size_max", "delta_pkt_size_mean",
    "delta_pkt_size_std", "delta_pkt_size_median", "delta_pkt_size_var",
    "delta_pkt_size_skew", "delta_pkt_size_cov", "delta_hdr_size_mean",
    "delta_hdr_size_std", "delta_hdr_size_max", "delta_pay_size_mean",
    "delta_pay_size_std", "delta_pay_size_max", "delta_pay_size_min",
    "delta_pkt_size_kurt", "delta_pkt_size_90th_percentile", "delta_payload_skew",
    "delta_payload_kurt", "iat_cov", "jitter_first_order", "jitter_second_order",
    "periodic_flow_flag", "iat_entropy", "rolling_pkt_count_cv_100ms",
    "windowed_95pct_rate", "peak_pkt_rate_1s", "std_iat", "iat_burst_ratio", "iat_mad",
    "flow_regularity_index", "silence_ratio", "peak_to_mean_iat_ratio", "burst_count",
    "mean_burst_length", "max_idle_gap", "burst_intensity", "burst_regularity",
    "burst_frequency", "active_period_mean", "idle_period_mean", "idle_period_max",
    "pkt_size_entropy", "fwd_packet_size_entropy", "rev_packet_size_entropy",
    "time_window_entropy_mean", "time_window_entropy_std", "fragment_size_entropy",
    "header_size_consistency", "pkt_size_25th_percentile", "pkt_size_75th_percentile",
    "pkt_size_90th_percentile", "iat_25th_percentile", "iat_75th_percentile",
    "iat_90th_percentile", "iat_range", "pkt_size_iqr", "iat_iqr",
    "payload_size_25th_percentile", "payload_size_75th_percentile",
    "payload_size_median", "header_size_median", "pkt_size_mad",
]

# Stage-1 uses UDP only; Stage-2 uses UDP + QUIC. The joint order is fixed so that
# UDP columns occupy indices 0..len(UDP_FEATURES)-1 in the Stage-2 matrix, which is
# exactly the Stage-1 feature index set the model escalates from.
STAGE1_FEATURES = list(UDP_FEATURES)
STAGE2_FEATURES = list(UDP_FEATURES) + list(QUIC_FEATURES)
STAGE1_INDICES = list(range(len(UDP_FEATURES)))  # UDP columns within the Stage-2 matrix


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    """Load the committed BCCC-UDP-QUIC sample and restrict to the closed-set families."""
    if not path.exists():
        raise FileNotFoundError(
            f"BCCC sample missing at {path}; it ships with the repo under data/sample/"
        )
    # numeric feature columns as float; keep the connection-key columns as strings.
    str_cols = set(CONN_KEYS + [TARGET])
    with gzip.open(path, "rt") as fh:
        header = fh.readline().strip().split(",")
    dtype = {c: (str if c in str_cols else np.float64) for c in header}
    df = pd.read_csv(path, dtype=dtype)
    df = df[df[TARGET].isin(FAMILIES)].reset_index(drop=True)
    return df


def connection_key(df: pd.DataFrame) -> np.ndarray:
    """Per-flow connection id = the UDP 5-tuple (protocol is constant). Two flows that
    share a connection must never straddle the train/test split."""
    return (
        df["src_ip"].astype(str) + "|" + df["src_port"].astype(str) + "|"
        + df["dst_ip"].astype(str) + "|" + df["dst_port"].astype(str)
    ).to_numpy()


def build_matrices(df: pd.DataFrame):
    """Return (X_stage2, y, groups). X_stage2 columns are STAGE2_FEATURES in order,
    so X_stage2[:, STAGE1_INDICES] is exactly the Stage-1 (UDP-only) matrix.
    Non-finite values are set to NaN and imputed later on the training split only."""
    X = df[STAGE2_FEATURES].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    y = df[TARGET].to_numpy()
    groups = connection_key(df)
    return X, y, groups


def leakage_safe_split(groups: np.ndarray, test_size: float = 0.25, seed: int = 42):
    """Connection-grouped split: no flow from one connection appears in both train and
    test. Returns (train_idx, test_idx) and asserts the no-shared-connection property."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
    shared = set(groups[train_idx]) & set(groups[test_idx])
    assert not shared, f"leakage: {len(shared)} connections in both splits"
    return train_idx, test_idx
