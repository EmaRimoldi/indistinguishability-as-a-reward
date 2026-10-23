from iar_phase0.generator import strip_code_fences


def test_strip_code_fences_prefers_implementation_over_examples() -> None:
    response = """
Explanation.

```
^[a-z]+_[a-z]+$
```

```python
def is_valid(value):
    return bool(value)
```
"""

    assert strip_code_fences(response) == "def is_valid(value):\n    return bool(value)"


def test_strip_code_fences_prefers_python_when_multiple_definitions_exist() -> None:
    response = """
```text
def pseudocode_example(value):
    several prose words
```

```python
def solve(value):
    return value + 1
```
"""

    assert strip_code_fences(response) == "def solve(value):\n    return value + 1"


def test_strip_code_fences_keeps_plain_code() -> None:
    response = "def solve(value):\n    return value + 1"

    assert strip_code_fences(response) == response
