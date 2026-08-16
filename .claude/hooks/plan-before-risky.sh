#!/usr/bin/env bash
# PreToolUse(Bash): nudge Claude to state its plan before consequential actions.
#
# Never blocks. Emits additionalContext, which the model reads before the command
# runs, so the cost of a false positive is one sentence — not a refused tool call.
#
# Triggers on the things that actually caused trouble in this repo:
#   - background jobs, which go silent for minutes and hide their cost
#   - git checkout -- / reset --hard / clean, which discard uncommitted work
#   - kill / pkill, which stop long fetches mid-write
#   - rm
set -uo pipefail

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -z "$cmd" ] && exit 0

hit=""
add() { hit="${hit:+$hit; }$1"; }

grep -qE '(^|[^[:alnum:]_])nohup([^[:alnum:]_]|$)' <<<"$cmd" && add "background job (nohup)"
grep -qE '&[[:space:]]*$'                          <<<"$cmd" && add "background job (trailing &)"
grep -qE 'git[[:space:]]+checkout[[:space:]]+--'   <<<"$cmd" && add "git checkout -- (discards uncommitted work)"
grep -qE 'git[[:space:]]+reset[[:space:]]+--hard'  <<<"$cmd" && add "git reset --hard"
grep -qE 'git[[:space:]]+clean'                    <<<"$cmd" && add "git clean"
grep -qE '(^|[;&|[:space:]])pkill([[:space:]]|$)'  <<<"$cmd" && add "pkill"
grep -qE '(^|[;&|[:space:]])kill([[:space:]]|$)'   <<<"$cmd" && add "kill"
grep -qE '(^|[;&|[:space:]])rm([[:space:]]|$)'     <<<"$cmd" && add "rm"

[ -z "$hit" ] && exit 0

jq -n --arg h "$hit" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    additionalContext: ("CONSEQUENTIAL COMMAND (" + $h + "). Before running it, say in one line: what it does, why now, and what you expect to change. If this is a re-run caused by your own bug, say that plainly. Do not go silent for minutes — the user needs the chance to interrupt before the cost is paid.")
  }
}'
exit 0
