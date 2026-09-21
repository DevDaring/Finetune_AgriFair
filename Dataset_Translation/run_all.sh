#!/usr/bin/env bash
# Full translation run: Hindi and Bengali in parallel, facts then advice within each language.
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a
PY=../.venv/bin/python
for lang in hi bn; do
  ( $PY -m Dataset_Translation.run translate --lang $lang --component agrifacts && \
    $PY -m Dataset_Translation.run translate --lang $lang --component agriadvice ) \
    > Dataset_Translation/out/run_$lang.log 2>&1 &
done
wait
$PY -m Dataset_Translation.run status > Dataset_Translation/out/run_status.txt 2>&1
touch Dataset_Translation/out/RUN_DONE
