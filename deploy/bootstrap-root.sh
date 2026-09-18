#!/bin/sh
set -eu

usage() {
  printf '%s\n' "Usage: sudo $0 --config PATH --model-key PATH [--exa-key PATH] [--source PATH] [--manual-source PATH]" >&2
  exit 2
}

CONFIG=
MODEL_KEY=
EXA_KEY=
SOURCE=/home/omega/llms/iterative_research_harness
MANUAL_SOURCE=/home/omega/llms/manual_intervention_tool
while test "$#" -gt 0; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --model-key) MODEL_KEY=$2; shift 2 ;;
    --exa-key) EXA_KEY=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --manual-source) MANUAL_SOURCE=$2; shift 2 ;;
    *) usage ;;
  esac
done
test -n "$CONFIG" && test -f "$CONFIG" || usage
test -n "$MODEL_KEY" && test -f "$MODEL_KEY" || usage
test -z "$EXA_KEY" || test -f "$EXA_KEY" || usage
test "$(id -u)" -eq 0 || { printf '%s\n' 'This one-time bootstrap must run as root.' >&2; exit 1; }
getent passwd rh-agent >/dev/null
getent group irh-worker >/dev/null
id -nG omega | tr ' ' '\n' | grep -qx irh-worker || usermod -aG irh-worker omega

# Remove only the known v0.1 deployment paths authorized by the operator.
systemctl disable --now irh-gpu-watchdog.timer 2>/dev/null || true
systemctl stop 'irh@*.service' 2>/dev/null || true
rm -f /etc/systemd/system/irh@.service \
  /etc/systemd/system/irh-gpu-watchdog.service \
  /etc/systemd/system/irh-gpu-watchdog.timer \
  /etc/sudoers.d/iterative-research-harness \
  /usr/local/bin/irh
rm -rf /opt/iterative-research-harness \
  /var/lib/iterative-research-harness \
  /home/rh-agent/.config/irh

install -d -o omega -g irh-worker -m 2775 /srv/irh /srv/irh/releases
install -d -o omega -g irh-worker -m 2775 /srv/manual-intervention-tool /srv/manual-intervention-tool/releases
install -d -o rh-agent -g irh-worker -m 2770 /var/lib/iterative-research-harness
runuser -u omega -- "$SOURCE/deploy/update.sh" --no-restart
runuser -u omega -- "$MANUAL_SOURCE/deploy/update.sh" --no-restart
install -o rh-agent -g irh-worker -m 0660 "$CONFIG" /var/lib/iterative-research-harness/config.json
install -o rh-agent -g irh-worker -m 0640 "$MODEL_KEY" /var/lib/iterative-research-harness/model.key
if test -n "$EXA_KEY"; then
  install -o rh-agent -g irh-worker -m 0640 "$EXA_KEY" /var/lib/iterative-research-harness/exa.key
fi
install -o root -g root -m 0644 "$SOURCE/deploy/irh-daemon.service" /etc/systemd/system/irh-daemon.service
ln -s /srv/irh/current/.venv/bin/irh /usr/local/bin/irh
systemctl daemon-reload
systemctl enable --now irh-daemon.service
printf '%s\n' 'Bootstrap complete. Future harness updates: deploy/update.sh'
