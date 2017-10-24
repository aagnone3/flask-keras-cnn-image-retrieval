#! /bin/bash

# vars
PYTEST_OPTIONS=-xs

# add the package's root directory to the PYTHONPATH
PYTHONPATH=${PYTHONPATH}:${TRAVIS_BUILD_DIR}

# invocation
echo ${PYTHONPATH}
echo ${PATH}
echo ${TRAVIS_BUILD_DIR}
py.test ${PYTEST_OPTIONS} test

