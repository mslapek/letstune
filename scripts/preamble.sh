set -e

if [ -z "$TEST_TENSORFLOW" ]; then
  export TEST_TENSORFLOW=0
fi

if [ -z "$TEST_XGBOOST" ]; then
  export TEST_XGBOOST=0
fi

python --version
