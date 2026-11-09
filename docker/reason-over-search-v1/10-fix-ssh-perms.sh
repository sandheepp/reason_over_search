#!/usr/bin/env bash
set -euo pipefail

# Vast/OpenSSH rejects keys if ownership/mode is too open.
# Ensure SSH key paths are always acceptable before auth attempts.
if [ -d /root/.ssh ]; then
  chown root:root /root/.ssh || true
  chmod 700 /root/.ssh || true
fi

if [ -f /root/.ssh/authorized_keys ]; then
  chown root:root /root/.ssh/authorized_keys || true
  chmod 600 /root/.ssh/authorized_keys || true
fi
