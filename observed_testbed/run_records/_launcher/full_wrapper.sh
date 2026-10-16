#!/usr/bin/env bash
# Runs the full 44-run launcher and records its exit status for the watcher.
cd "$(dirname "$0")/../.."
echo '{"status":"running"}' > runs/_launcher/full_status.json
bash testbed/run_all.sh -P 16 > runs/_launcher/full_launch.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -f runs/ALL_DONE ]; then s=done; else s=failed; fi
printf '{"status":"%s","exit_code":%d}\n' "$s" "$rc" > runs/_launcher/full_status.json
