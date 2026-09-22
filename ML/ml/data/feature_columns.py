"""
Shared constants for the NSL-KDD dataset.

NSL-KDD has 41 features + 1 label column (+ a difficulty column in the
raw files, which we drop). This module is imported by both the training
pipeline and (eventually) the live-capture feature extractor, so that
both sides agree on column names, categorical columns, and how raw
attack names map to the 5 coarse attack categories used for classification.
"""

# Ordered list of the 41 NSL-KDD feature names, in the order they appear
# in the raw KDDTrain+.txt / KDDTest+.txt files.
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Full column list for the raw files: 41 features + label + difficulty score
RAW_COLUMNS = FEATURE_COLUMNS + ["label", "difficulty"]

# Categorical columns that need encoding before any classical ML model
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Numeric columns = everything else in FEATURE_COLUMNS
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

# NSL-KDD has dozens of specific attack labels. For a manageable multi-class
# classification task we bucket them into the 5 standard coarse categories.
ATTACK_CATEGORY_MAP = {
    "normal": "normal",

    # DoS (Denial of Service)
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",

    # Probe (surveillance / scanning)
    "satan": "probe", "ipsweep": "probe", "nmap": "probe",
    "portsweep": "probe", "mscan": "probe", "saint": "probe",

    # R2L (Remote to Local — unauthorized remote access)
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l",
    "phf": "r2l", "multihop": "r2l", "warezmaster": "r2l",
    "warezclient": "r2l", "spy": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "snmpguess": "r2l", "snmpgetattack": "r2l", "httptunnel": "r2l",
    "sendmail": "r2l", "named": "r2l",

    # U2R (User to Root — privilege escalation)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}

# The 5 target classes, in a fixed order used for label encoding everywhere
ATTACK_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]


def map_label_to_category(raw_label: str) -> str:
    """Map a raw NSL-KDD label (e.g. 'neptune', 'smurf') to its coarse
    attack category. Unknown labels (can appear in KDDTest+ as novel
    attacks not seen in training) fall back to 'r2l' as the closest
    generic 'unauthorized access' bucket rather than raising."""
    key = raw_label.strip().rstrip(".").lower()
    return ATTACK_CATEGORY_MAP.get(key, "r2l")
