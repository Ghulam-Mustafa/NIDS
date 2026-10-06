"""Column-name normalisation so CICIDS2017/2018 files and CICFlowMeter outputs all map to one schema."""
import re

_F17 = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
        "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
        "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]

# CSE-CIC-IDS2018 / CICFlowMeter-V4-style abbreviations, in the same order as _F17.
_F18 = ["Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
        "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
        "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
        "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
        "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
        "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
        "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
        "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
        "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
        "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
assert len(_F17) == len(_F18)


def _key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_ALIASES = {}
for _c17, _c18 in zip(_F17, _F18):
    for _a in (_c17, _c18):
        k = _key(_a)
        for v in (k, k.replace("packets", "packet"), k.replace("backward", "bwd")):
            _ALIASES.setdefault(v, _c17)

# Identifier columns -> one spelling (they are dropped before modelling but returned by `predict`).
for _canon, _names in {"Flow ID": ["flowid"], "Src IP": ["srcip", "sourceip"],
                       "Dst IP": ["dstip", "destinationip"], "Src Port": ["srcport", "sourceport"],
                       "Timestamp": ["timestamp"], "Protocol": ["protocol"]}.items():
    for _n in _names:
        _ALIASES[_n] = _canon


def normalize_columns(df):
    """Rename any known spelling (2017, 2018, snake_case ports) to the CICIDS2017 name."""
    names = [_ALIASES.get(_key(c), str(c).strip()) for c in df.columns]
    df = df.copy(deep=False)
    df.columns = names
    return df.loc[:, ~df.columns.duplicated()]
