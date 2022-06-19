#!/usr/bin/env dash

. scripts/preamble.sh

## mypy ##

mypy_args="letstune tests examples/sklearn"

if [ "$TEST_TENSORFLOW" -ne 0 ]; then
  mypy_args="${mypy_args} examples/keras"
fi

if [ "$TEST_XGBOOST" -ne 0 ]; then
  mypy_args="${mypy_args} examples/xgboost"
fi

## end mypy ##

set -x

isort --check letstune tests examples
black --check letstune tests examples
flake8 letstune tests examples
mypy $mypy_args

echo "Lint OK!"
