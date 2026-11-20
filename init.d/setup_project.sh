#!/bin/bash
# Renku session start: install the workspace packages from the session's repo copy and expose
# openai_api_key from the mounted data connector as OPENAI_API_KEY. Log: ~/setup_project.log
_kk_main() {
  local log="$HOME/setup_project.log" root="${RENKU_WORKING_DIR:-/home/renku/work}"
  local venv="${UV_PROJECT_ENVIRONMENT:-$root/.venv}" repo key_file py i
  _kk_log() { echo "[$(date +%T)] $*" >> "$log"; }

  # The image's editable installs point at the build directory; reinstall from the session copy.
  repo=$(dirname "$(find "$root" -maxdepth 3 -path '*/conversational-toolkit/pyproject.toml' 2>/dev/null | head -n1)")
  repo=$(dirname "$repo")
  py="$venv/bin/python"
  [ -x "$py" ] || py=$(command -v python)
  local -a install=("$py" -m pip install)
  command -v uv > /dev/null && install=(uv pip install --python "$py")
  if [ -d "$repo/backend" ]; then
    "${install[@]}" --no-deps -e "$repo/conversational-toolkit" -e "$repo/backend" >> "$log" 2>&1 \
      && _kk_log "installed packages from $repo into $py" || _kk_log "package install FAILED ($repo, $py)"
  else
    _kk_log "repo not found under $root"
  fi

  # The data connector can take a few seconds to mount.
  for i in $(seq 1 20); do
    key_file=$(find "$root" -maxdepth 3 -type f \( -name openai_api_key -o -name openai_api_key.txt \) 2>/dev/null | head -n1)
    [ -n "$key_file" ] && break
    sleep 1
  done
  if [ -z "$key_file" ]; then _kk_log "no openai_api_key found under $root"; return 0; fi

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
  _kk_log "OPENAI_API_KEY set up from $key_file"
}
_kk_main
unset -f _kk_main _kk_log
