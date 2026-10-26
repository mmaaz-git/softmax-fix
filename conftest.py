"""Replay saved examples by default; use --search for fresh Hypothesis cases."""

import pytest
from hypothesis import Phase, settings

settings.register_profile("replay", phases=[Phase.explicit], database=None)
settings.register_profile(
    "search",
    phases=[Phase.generate, Phase.target, Phase.shrink, Phase.explain],
    database=None,
)


def pytest_addoption(parser):
    parser.addoption(
        "--search",
        action="store_true",
        help="Search for new counterexamples instead of replaying saved examples.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Select the profile before Hypothesis processes its command-line options.
    if config.getoption("search"):
        config.option.hypothesis_profile = "search"
    elif config.getoption("hypothesis_profile") is None:
        config.option.hypothesis_profile = "replay"
