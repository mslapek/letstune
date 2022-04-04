#!/usr/bin/env dash
set -ex

if [ -z "$TEST_TENSORFLOW" ]; then
  export TEST_TENSORFLOW=0
fi

isort --check letstune tests examples
black --check letstune tests examples
mypy letstune tests examples
flake8 letstune tests examples

## pytest ##

pytest_args="--doctest-modules --ignore=examples"

if [ "$TEST_TENSORFLOW" -eq 0 ]; then
  pytest_args="${pytest_args} --ignore=tests/test_keras.py --ignore=letstune/keras.py"
fi

pytest $pytest_args

## pytest examples ##

examples_to_test="examples/sklearn/*.py"

if [ "$TEST_TENSORFLOW" -ne 0 ]; then
  examples_to_test="${examples_to_test} examples/keras/*.py"
fi

pytest $examples_to_test

echo "Lint OK!"
