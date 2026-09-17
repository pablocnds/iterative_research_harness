#!/bin/sh
set -eu

SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RELEASE_ROOT=${IRH_RELEASE_ROOT:-/srv/irh/releases}
CURRENT_LINK=${IRH_CURRENT_LINK:-/srv/irh/current}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REVISION=$(git -C "$SOURCE_DIR" rev-parse --short=12 HEAD 2>/dev/null || printf 'uncommitted')
SOURCE_SHA=$(find "$SOURCE_DIR/src" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
RELEASE="$RELEASE_ROOT/$STAMP-$REVISION"
UV_BIN=${UV_BIN:-/home/omega/.local/bin/uv}
DEPLOY_PYTHON=${DEPLOY_PYTHON:-/usr/bin/python3}

test "$(id -u)" -ne 0 || { printf '%s\n' 'Run deploy/update.sh as omega, not root.' >&2; exit 1; }
test -d "$RELEASE_ROOT" || { printf '%s\n' 'Run the one-time bootstrap first.' >&2; exit 1; }
test ! -e "$RELEASE"
mkdir "$RELEASE"
cleanup() { find "$RELEASE" -depth -delete 2>/dev/null || true; }
trap cleanup EXIT HUP INT TERM
cp -a "$SOURCE_DIR/src" "$SOURCE_DIR/pyproject.toml" "$SOURCE_DIR/README.md" "$SOURCE_DIR/uv.lock" "$RELEASE/"
printf '{"version":"0.2.0","git_revision":"%s","source_sha256":"%s"}\n' \
  "$REVISION" "$SOURCE_SHA" > "$RELEASE/release.json"
"$UV_BIN" sync --directory "$RELEASE" --frozen --no-dev --python "$DEPLOY_PYTHON"
chmod -R go-w "$RELEASE"
SWITCH_LINK=$(dirname "$CURRENT_LINK")/.current-$STAMP
ln -s "$RELEASE" "$SWITCH_LINK"
mv -Tf "$SWITCH_LINK" "$CURRENT_LINK"
trap - EXIT HUP INT TERM

if test "${1:-}" != "--no-restart" && test -S /var/lib/iterative-research-harness/daemon.sock; then
  "$CURRENT_LINK/.venv/bin/irh" restart-daemon
fi
printf 'Installed %s\n' "$RELEASE"
