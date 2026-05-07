#!/bin/sh
set -eu

if [ -z "${FATTERVOICE_MODEL:-}" ]; then
  if [ "${MODEL_SELECTION_HINT:-omnivoice}" = "all" ]; then
    export FATTERVOICE_MODEL="omnivoice"
  else
    export FATTERVOICE_MODEL="${MODEL_SELECTION_HINT:-omnivoice}"
  fi
fi

exec "$@"
