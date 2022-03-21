#!/usr/bin/env bash
set -ex

isort --check letstune tests
black --check letstune tests
mypy letstune tests
flake8 letstune tests
pytest --doctest-modules

echo "Lint OK!"
