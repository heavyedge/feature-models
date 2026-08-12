heartbeat_dir="${HEARTBEAT_DIR:-/heartbeat}"

current_step() {
  if [ -s "$heartbeat_dir/current-step" ]; then
    head -n 1 "$heartbeat_dir/current-step"
  else
    printf '%s\n' "Build container"
  fi
}

finalize_check() {
  bash -euo pipefail .github/scripts/check-run.sh "$@"
  touch "$heartbeat_dir/check-finalized"
}

if [ -f "$heartbeat_dir/check-finalized" ]; then
  exit 0
elif [ -f "$heartbeat_dir/build-cancelled" ]; then
  step_name="$(current_step)"
  finalize_check completed cancelled "Job was cancelled" \
    "Job was cancelled during: ${step_name}."
elif [ -f "$heartbeat_dir/build-failure" ]; then
  step_name="$(current_step)"
  failure_summary="${step_name} failed."
  if [ -s "$heartbeat_dir/failure-exit-code" ]; then
    exit_code="$(head -n 1 "$heartbeat_dir/failure-exit-code")"
    failure_summary="${step_name} exited with status ${exit_code}."
  fi
  finalize_check completed failure "${step_name} failed" "$failure_summary"
elif [ -f "$heartbeat_dir/build-success" ]; then
  finalize_check completed success "Build completed successfully"
elif [ -f "$heartbeat_dir/build-started" ]; then
  heartbeat_file="$heartbeat_dir/heartbeat"
  heartbeat_source="$heartbeat_dir/build-started"
  if [ -f "$heartbeat_file" ]; then
    heartbeat_source="$heartbeat_file"
  fi

  heartbeat_age="$(($(date +%s) - $(stat -c %Y "$heartbeat_source")))"
  if [ "$heartbeat_age" -gt 90 ]; then
    step_name="$(current_step)"
    finalize_check completed failure "${step_name} stopped responding" \
      "No build-container heartbeat was received for ${heartbeat_age} seconds during: ${step_name}."
  fi
fi
