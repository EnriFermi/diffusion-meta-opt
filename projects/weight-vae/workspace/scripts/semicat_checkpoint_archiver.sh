#!/usr/bin/env bash
set -Eeuo pipefail

# Archive every new version of a live SemiCat last checkpoint.
# Override these variables when launching, e.g.
#   SOURCE_CHECKPOINT=/path/to/last.ckpt ARCHIVE_DIR=/path/to/archive ./...sh

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/home/coder/project/external/semicat_fresh_20260711/logs/train/2026-07-11_00-36-40_12345_local/checkpoints/last.ckpt}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/home/coder/project/artifacts/semicat_checkpoint_archive_20260711}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
LOG_FILE="${LOG_FILE:-${ARCHIVE_DIR}/archiver.log}"

mkdir -p "$ARCHIVE_DIR"
exec >>"$LOG_FILE" 2>&1

echo "[$(date -Is)] archiver_started source=$SOURCE_CHECKPOINT archive=$ARCHIVE_DIR interval_sec=$INTERVAL_SEC"

last_signature=""
while true; do
    if [[ -f "$SOURCE_CHECKPOINT" ]]; then
        size="$(stat -c '%s' "$SOURCE_CHECKPOINT")"
        mtime="$(stat -c '%Y' "$SOURCE_CHECKPOINT")"
        signature="${size}:${mtime}"

        if [[ "$signature" != "$last_signature" && "$size" -gt 0 ]]; then
            sha="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
            short_sha="${sha:0:16}"

            # A restarted watcher must not duplicate an already archived blob.
            existing="$(find "$ARCHIVE_DIR" -maxdepth 1 -type f -name "last_*_${short_sha}.ckpt" -print -quit)"
            if [[ -n "$existing" ]]; then
                echo "[$(date -Is)] already_archived existing=$existing sha256=$sha"
                last_signature="$signature"
                sleep "$INTERVAL_SEC"
                continue
            fi

            timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
            destination="${ARCHIVE_DIR}/last_${timestamp}_${short_sha}.ckpt"
            temporary="${destination}.partial"

            # Copy atomically so a consumer never sees a partial checkpoint.
            cp -p --reflink=auto "$SOURCE_CHECKPOINT" "$temporary"
            mv -f "$temporary" "$destination"
            {
                echo "source=$SOURCE_CHECKPOINT"
                echo "destination=$destination"
                echo "size_bytes=$size"
                echo "source_mtime_epoch=$mtime"
                echo "sha256=$sha"
                echo "copied_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            } >"${destination%.ckpt}.txt"

            echo "[$(date -Is)] archived destination=$destination size_bytes=$size sha256=$sha"
            last_signature="$signature"
        fi
    else
        echo "[$(date -Is)] waiting_for_source source=$SOURCE_CHECKPOINT"
    fi

    sleep "$INTERVAL_SEC"
done
