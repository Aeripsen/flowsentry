"""
NSL-KDD loader + preprocessing.

NSL-KDD is used here ONLY as the common public benchmark, so the model can be
reproduced by anyone and compared against the baseline everyone uses. It is NOT
the headline dataset: the headline system runs on flows produced by Sepehr's own
UDPFlowLyzer / QUICFlowLyzer extractors (Week-2 milestone). See docs/MODEL_CARD.md.

NSL-KDD ships as separate KDDTrain+ / KDDTest+ files whose test set contains
attack types absent from train, so it is a leakage-safe, novelty-aware split by
construction. We fit the preprocessor on train only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]
DROP = ["label", "difficulty"]

# Standard NSL-KDD attack-name -> 5-class category mapping.
ATTACK_CATEGORY = {
    "normal": "normal",
    # DoS
    "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "mailbomb": "dos", "apache2": "dos", "processtable": "dos",
    "udpstorm": "dos", "worm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["category"] = df["label"].map(ATTACK_CATEGORY).fillna("unknown_attack")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric = [c for c in COLUMNS if c not in CATEGORICAL + DROP]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", numeric),
        ]
    )


def load(data_dir: Path = DATA_DIR):
    """Return (X_train, y_train, X_test, y_test, preprocessor, feature_names)."""
    train = _read(data_dir / "KDDTrain+.txt")
    test = _read(data_dir / "KDDTest+.txt")

    pre = build_preprocessor()
    X_train = pre.fit_transform(train)
    X_test = pre.transform(test)
    feature_names = list(pre.get_feature_names_out())
    y_train = train["category"].to_numpy()
    y_test = test["category"].to_numpy()
    return X_train, y_train, X_test, y_test, pre, feature_names
