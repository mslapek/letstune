#!/usr/bin/env dash

. scripts/preamble.sh

if [ -d ~/ltexamples ]; then
  rm -R ~/ltexamples
fi

examples="examples/sklearn/*.py"

if [ "$TEST_TENSORFLOW" -ne 0 ]; then
  examples="${examples} examples/keras/*.py"
fi

if [ "$TEST_XGBOOST" -ne 0 ]; then
  examples="${examples} examples/xgboost/*.py"
fi

export PYTHONPATH=$(pwd)

for example in $examples; do
  printf "== First run of %s ==\n" "$example"
  time python3 "$example"
  printf "\n"

  printf "== Second run of %s ==\n" "$example"
  # second run is to test experiment restoring from letstune.db
  time python3 "$example"
  printf "\n\n"
done


echo "Examples OK!"
