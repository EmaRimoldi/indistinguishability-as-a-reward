"""Pinned denotation comparison from the official Spider test-suite evaluator.

This module is a security-hardened derivative of ``exec_eval.py`` from
``taoyds/test-suite-sql-eval``.  Query execution is deliberately not copied:
the upstream implementation opens SQLite normally and its timeout runs in the
same process.  :mod:`iar_spider.evaluate_validation` supplies a read-only,
process-isolated executor while retaining the upstream result-equivalence
semantics.

Upstream: https://github.com/taoyds/test-suite-sql-eval
Revision: e97acc546ecbee8fa27fa8dbf025ef61493a876c
Source SHA-256 (exec_eval.py):
29d034db28904490c28037537a14fbb0150b6e86cef0049076c0511d6b6b77f7
License: MIT; see ``reproduction/third_party/test-suite-sql-eval/LICENSE``.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from itertools import product
from typing import Any

import sqlparse

OFFICIAL_TEST_SUITE_REPOSITORY = "https://github.com/taoyds/test-suite-sql-eval.git"
OFFICIAL_TEST_SUITE_REVISION = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"
OFFICIAL_EXEC_EVAL_SHA256 = (
    "29d034db28904490c28037537a14fbb0150b6e86cef0049076c0511d6b6b77f7"
)
OFFICIAL_LICENSE_SHA256 = (
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)


def postprocess(query: str) -> str:
    """Apply the three whitespace repairs used by upstream ``exec_eval.py``."""
    return query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")


def remove_distinct(query: str) -> str:
    """Mirror the official evaluator's token-level DISTINCT removal."""
    statements = sqlparse.parse(query)
    if len(statements) != 1:
        raise ValueError("DISTINCT normalization requires one SQL statement")
    tokens = [token.value for token in statements[0].flatten()]
    return "".join(token for token in tokens if token.lower() != "distinct")


def replace_current_year(query: str) -> str:
    """Mirror the upstream CURDATE-year compatibility rewrite."""
    return re.sub(
        r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*",
        "2020",
        query,
        flags=re.IGNORECASE,
    )


def _permute_tuple(element: Sequence[Any], permutation: Sequence[int]) -> tuple[Any, ...]:
    assert len(element) == len(permutation)
    return tuple(element[index] for index in permutation)


def _unorder_row(row: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sorted(row, key=lambda value: str(value) + str(type(value))))


def _quick_reject(
    first: Sequence[Sequence[Any]],
    second: Sequence[Sequence[Any]],
    *,
    order_matters: bool,
) -> bool:
    unordered_first = [_unorder_row(row) for row in first]
    unordered_second = [_unorder_row(row) for row in second]
    if order_matters:
        return unordered_first == unordered_second
    return set(unordered_first) == set(unordered_second)


def _multiset_equal(first: Iterable[Any], second: Iterable[Any]) -> bool:
    first_values = list(first)
    second_values = list(second)
    if len(first_values) != len(second_values):
        return False
    counts: defaultdict[Any, int] = defaultdict(int)
    for value in first_values:
        counts[value] += 1
    for value in second_values:
        counts[value] -= 1
        if counts[value] < 0:
            return False
    return True


def _constraint_permutations(
    first_sets_by_column: list[set[Any]],
    second: Sequence[Sequence[Any]],
    *,
    rng: random.Random,
) -> Iterator[tuple[int, ...]]:
    column_count = len(second[0])
    constraints = [set(range(column_count)) for _ in range(column_count)]
    if column_count <= 3:
        return product(*constraints)
    for _ in range(20):
        sampled_second_row = rng.choice(second)
        for first_column in range(column_count):
            for second_column in set(constraints[first_column]):
                if sampled_second_row[second_column] not in first_sets_by_column[first_column]:
                    constraints[first_column].remove(second_column)
    return product(*constraints)


def result_equal(
    first: list[tuple[Any, ...]],
    second: list[tuple[Any, ...]],
    *,
    order_matters: bool,
    seed: int = 0,
) -> bool:
    """Return official test-suite denotation equality, deterministically seeded."""
    if not first and not second:
        return True
    if len(first) != len(second):
        return False
    column_count = len(first[0])
    if len(second[0]) != column_count:
        return False
    if any(len(row) != column_count for row in first + second):
        return False
    if not _quick_reject(first, second, order_matters=order_matters):
        return False

    first_sets_by_column = [
        {row[column] for row in first} for column in range(column_count)
    ]
    for permutation in _constraint_permutations(
        first_sets_by_column, second, rng=random.Random(seed)
    ):
        if len(permutation) != len(set(permutation)):
            continue
        permuted_second = (
            second
            if column_count == 1
            else [_permute_tuple(row, permutation) for row in second]
        )
        if order_matters:
            if first == permuted_second:
                return True
        elif set(first) == set(permuted_second) and _multiset_equal(
            first, permuted_second
        ):
            return True
    return False
