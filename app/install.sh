#!/usr/bin/env bash
# Kubernetes only: builds the Python virtualenv on the persistent volume.
# Skips the work when the venv already matches requirements.txt, so only the first start
# (or a start after requirements.txt changed) takes long, 5-15 min.
set -euo pipefail
VENV="${VENV:-/state/venv}"
REQ="${REQ:-/app/requirements.txt}"
want=$(sha256sum "$REQ" | cut -d' ' -f1)
if [ -f "$VENV/.requirements-sha" ] && [ "$(cat "$VENV/.requirements-sha")" = "$want" ]; then
  echo "virtualenv is up to date"
  exit 0
fi
echo "building virtualenv in $VENV ..."
rm -rf "$VENV"
python -m venv "$VENV"
"$VENV/bin/pip" install --no-cache-dir --disable-pip-version-check -r "$REQ"
echo "$want" > "$VENV/.requirements-sha"
