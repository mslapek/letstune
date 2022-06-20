#!/usr/bin/env dash

. scripts/preamble.sh

pytest_args="--ignore=examples"

if [ "$TEST_TENSORFLOW" -eq 0 ]; then
  pytest_args="${pytest_args} --ignore=tests/test_keras.py --ignore=letstune/keras.py"
fi

set -x

pytest $pytest_args

echo "Tests OK!"
