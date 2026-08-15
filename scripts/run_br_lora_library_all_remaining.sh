#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------
# Require the BR-LoRA Conda environment.
# ------------------------------------------------------------

if [[ "${CONDA_DEFAULT_ENV:-}" != "brats-nnunet" ]]; then
    echo "ERROR: required Conda environment is not active." >&2
    echo "Expected: brats-nnunet" >&2
    echo "Observed: ${CONDA_DEFAULT_ENV:-<none>}" >&2
    echo >&2
    echo "Run:" >&2
    echo "  conda activate brats-nnunet" >&2
    echo "and then restart this script." >&2
    exit 1
fi

PROJECT_ROOT="$HOME/Desktop/localized-medical-image-synthesis"
LOG_ROOT="$PROJECT_ROOT/logs/br_lora_library_pipeline"

START_BATCH=7
END_BATCH=40

mkdir -p "$LOG_ROOT"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "BR-LoRA REMAINING LIBRARY PRODUCTION"
echo "============================================================"
echo "Conda env   : ${CONDA_DEFAULT_ENV}"
echo "Python      : $(which python)"
echo "Start batch : $(printf 'batch_%04d' "$START_BATCH")"
echo "End batch   : $(printf 'batch_%04d' "$END_BATCH")"
echo "Log root    : $LOG_ROOT"
echo

for n in $(seq "$START_BATCH" "$END_BATCH"); do

    BATCH="$(printf 'batch_%04d' "$n")"
    LOG="$LOG_ROOT/${BATCH}.log"

    echo
    echo "============================================================"
    echo "STARTING $BATCH"
    echo "============================================================"
    echo "Log: $LOG"
    echo

    {
        echo "============================================================"
        echo "BR-LoRA PIPELINE: $BATCH"
        echo "Started: $(date)"
        echo "============================================================"
        echo

        python \
          scripts/run_br_lora_library_pipeline.py \
          --batch "$BATCH" \
          --accept \
          --cleanup

        echo
        echo "============================================================"
        echo "COMPLETED: $BATCH"
        echo "Finished: $(date)"
        echo "============================================================"

    } 2>&1 | tee "$LOG"

    PIPE_STATUS=${PIPESTATUS[0]}

    if [ "$PIPE_STATUS" -ne 0 ]; then
        echo
        echo "============================================================"
        echo "PIPELINE STOPPED"
        echo "============================================================"
        echo "Failed batch : $BATCH"
        echo "Log          : $LOG"
        echo
        echo "No later batches were started."
        exit "$PIPE_STATUS"
    fi

    echo
    echo "$BATCH completed successfully."

done

echo
echo "============================================================"
echo "ALL REQUESTED BR-LoRA BATCHES COMPLETE"
echo "============================================================"
echo "Final batch : $(printf 'batch_%04d' "$END_BATCH")"
echo "Finished    : $(date)"
echo "============================================================"
