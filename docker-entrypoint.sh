#!/bin/sh
set -eu

if [ -z "${FATTERQWEN_MODEL:-}" ]; then
  if [ "${MODEL_SELECTION_HINT:-1.7B}" = "all" ]; then
    export FATTERQWEN_MODEL="1.7B"
  else
    export FATTERQWEN_MODEL="${MODEL_SELECTION_HINT:-1.7B}"
  fi
fi

exec "$@"
