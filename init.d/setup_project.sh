#!/bin/bash
# Renku session start: make the OpenAI key from the mounted data connector (a PolyBox or
# SwitchDrive folder containing a file named openai_api_key or openai_api_key.txt) available as OPENAI_API_KEY in
# terminals and notebook kernels. The key is read at each start and never copied.
_kk_main() {
  local log="$HOME/setup_project.log" root="${RENKU_WORKING_DIR:-/home/renku/work}" key_file="" i
  _kk_log() { echo "[$(date +%T)] $*" >> "$log"; }

  # The data connector can take a few seconds to mount.
  for i in $(seq 1 20); do
    key_file=$(find "$root" -maxdepth 3 -type f \( -name openai_api_key -o -name openai_api_key.txt \) 2>/dev/null | head -n1)
    [ -n "$key_file" ] && break
    sleep 1
  done
  if [ -z "$key_file" ]; then _kk_log "no openai_api_key found under $root, skipping"; return 0; fi
  _kk_log "key file: $key_file"

  local line="if [ -z \"\${OPENAI_API_KEY:-}\" ] && [ -r \"$key_file\" ]; then export OPENAI_API_KEY=\"\$(tr -d '[:space:]' < \"$key_file\")\"; fi"
  touch "$HOME/.bashrc"
  grep -Fqx "$line" "$HOME/.bashrc" || printf '\n%s\n' "$line" >> "$HOME/.bashrc"

  mkdir -p "$HOME/.ipython/profile_default/startup"
  cat > "$HOME/.ipython/profile_default/startup/00-openai-key.py" <<PY
import os

if "OPENAI_API_KEY" not in os.environ and os.path.exists("$key_file"):
    with open("$key_file") as f:
        os.environ["OPENAI_API_KEY"] = "".join(f.read().split())
PY
  _kk_log "OPENAI_API_KEY set up for bash and IPython"
}
_kk_main
unset -f _kk_main _kk_log
