"""Decorators that run one check against both the PR code and the reference fix.

    @bug("01_decorators/L1_timing")                   # must fail on the PR code, pass on the fix
    @bug("01_decorators/L1_timing", raises=TypeError) # ...and fail with this exception type
    @fine("01_decorators/L1_timing")                  # red herring: must pass on both
    def test_L1_something(m):
        ...

A `bug` check is a strict xfail on the PR side, so the suite goes red if a planted
bug doesn't actually reproduce, or if it fails for a different reason than claimed.
"""
import pytest


def bug(drill: str, raises: type[BaseException] | tuple[type[BaseException], ...] = AssertionError):
    return pytest.mark.parametrize(
        "m",
        [
            pytest.param(
                ("pr", drill),
                id="pr",
                marks=pytest.mark.xfail(strict=True, raises=raises, reason="planted bug"),
            ),
            pytest.param(("fixed", drill), id="fixed"),
        ],
        indirect=True,
    )


def fine(drill: str):
    return pytest.mark.parametrize(
        "m",
        [pytest.param(("pr", drill), id="pr"), pytest.param(("fixed", drill), id="fixed")],
        indirect=True,
    )
