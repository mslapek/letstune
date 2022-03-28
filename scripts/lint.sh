#!/usr/bin/env bash
set -ex

isort --check letstune tests examples
black --check letstune tests examples
mypy letstune tests examples
flake8 letstune tests examples
pytest --doctest-modules

pytest examples/*/digits_*.py

echo "Lint OK!"
