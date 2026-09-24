from collections import Counter

BINARY_CLASSES = ["Normal", "Adventitious"]
COARSE_CLASSES = ["Normal", "CAS only", "DAS only", "CAS and DAS"]
CHALLENGE_CLASSES = {
    "T2-1": ["Normal", "Adventitious", "Poor Quality"],
    "T2-2": ["Normal", "CAS", "DAS", "CAS & DAS", "Poor Quality"],
}
CHALLENGE_SENSITIVITY = {
    "T2-1": ["Adventitious"],
    "T2-2": ["CAS", "DAS", "CAS & DAS"],
}


def counts(rows, key):
    return Counter(row[key] for row in rows)


def assert_exclusion_contract(rows):
    reasons = Counter(row["exclusion_reason"] for row in rows if row["inclusion_status"] == "excluded")
    assert reasons["poor_quality_only"] == 230
    assert reasons["exact_duplicate_of_heldout_record"] == 1
    assert sum(reasons.values()) == 231

