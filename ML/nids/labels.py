"""Label harmonisation for CICIDS2017 and CSE-CIC-IDS2018."""

LABEL_COL = "Label"
BENIGN = "BENIGN"

# Order matters: more specific rules first ("ddos" before "dos").
_RULES = [
    (("web attack", "xss", "sql injection", "brute force -web", "brute force -xss"), "WebAttack"),
    (("brute", "patator"), "BruteForce"),        # FTP/SSH-Patator, FTP/SSH-Bruteforce
    (("ddos",), "DDoS"),
    (("dos",), "DoS"),                           # Hulk, GoldenEye, slowloris, Slowhttptest
    (("portscan",), "PortScan"),
    (("bot",), "Bot"),
    (("infiltr",), "Infiltration"),
    (("heartbleed",), "Heartbleed"),
    (("benign",), BENIGN),
]


def map_label(raw) -> str:
    s = str(raw).lower()
    for keys, name in _RULES:
        if any(k in s for k in keys):
            return name
    return "Other"
