from collections import Counter

from iar_campaign.data_protocol import grouped_stratified_partition


def test_grouped_stratification_is_exact_nested_safe_and_deterministic():
    rows = []
    for index in range(60):
        group = f"g{index // 2}" if index < 12 else f"g{index}"
        rows.append(
            {
                "id": f"row-{index}",
                "group": group,
                "stratum": "a" if index % 3 else "b",
            }
        )
    sizes = {"probe": 10, "pilot_delta": 10, "core_delta": 10, "validation": 10, "reserve": 20}
    first, first_audit = grouped_stratified_partition(
        rows,
        role_sizes=sizes,
        group_key=lambda row: row["group"],
        stratum_key=lambda row: row["stratum"],
        seed=23,
    )
    second, second_audit = grouped_stratified_partition(
        rows,
        role_sizes=sizes,
        group_key=lambda row: row["group"],
        stratum_key=lambda row: row["stratum"],
        seed=23,
    )
    assert {role: len(values) for role, values in first.items()} == sizes
    assert first == second
    assert first_audit == second_audit
    group_roles = {}
    for role, values in first.items():
        for row in values:
            group_roles.setdefault(row["group"], set()).add(role)
    assert all(len(roles) == 1 for roles in group_roles.values())
    source_counts = Counter(row["stratum"] for row in rows)
    assert source_counts == Counter(
        row["stratum"] for values in first.values() for row in values
    )
