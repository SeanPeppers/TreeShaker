#!/bin/bash -eu
pip3 install --require-hashes --no-deps -r .clusterfuzzlite/requirements.txt
pip3 install --no-deps .
for fuzzer in $(find fuzz -name 'fuzz_*.py'); do
  compile_python_fuzzer "$fuzzer"
done
