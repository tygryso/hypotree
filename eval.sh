#!/bin/bash
#
# Phase-4 evaluation gate runner.
#
#   0. sync + lint + format + test
#   1. generate the held-out landscapes + briefings
#   2. pre-flight: assert the analytic difficulty calibration holds
#   2b. pre-flight: assert the ENGINE can solve every seed with no LLM at all
#   3. criterion 2 — TS-quality ablation (no LLM)
#   4. criteria 1/3/4 — three paired arms on every seed
#   5. score the frozen gate → GO / STOP / ITERATE
#   6. render the human/LLM-readable markdown report
#
# Three arms run per seed, because "does a belief state help?" is two questions:
#   B  hypotree belief-state tools
#   F  automatic flat experiment log + notes  → criterion 1b (informational moat)
#   A  manual scratchpad only                → criterion 1a (ergonomic moat)
# The gate is decided by 1b; 1a is reported alongside it.
#
# The seed set and arm list live in eval/runner/config.py — never duplicated
# here, so the pre-registered protocol cannot drift from the harness.
#
# Every invocation is namespaced by a run-id (either explicitly via --run-id, or
# auto-generated via --run-iteration). Logs land in eval/runs/<run-id>/ and the
# belief-state DBs in the <run-id>@hypotree-eval workspace, so two runs can
# never contaminate each other's results.

set -euo pipefail

HTTP_PORT=8080
LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL="qwen3.6:27b-q8_0"
RUN_ID=""
RUN_ITERATION=""
BRANCH_OVERRIDE=""
SKIP_CHECKS=0
RESUME=0

print_usage() {
    echo "Usage: $0 --run-id ID | --run-iteration LETTER [options]"
    echo "Run Identification (choose one):"
    echo "  --run-id VAL           Isolates logs, workspace and belief-state DBs"
    echo "  --run-iteration VAL    Auto-generates run-id as:"
    echo "                         v{version}_run-iteration~{VAL}_llm-model~{MODEL}[_branch~{BRANCH}]"
    echo "                         Version is fetched via Python. Branch is fetched"
    echo "                         from git (or --branch), and omitted if empty."
    echo "Options:"
    echo "  --http-port VAL        (Default: ${HTTP_PORT})"
    echo "  --llm-base-url URI     (Default: ${LLM_BASE_URL})"
    echo "  --llm-model VAL        (Default: ${LLM_MODEL})"
    echo "  --branch VAL           Override git branch detection (useful in Docker)"
    echo "  --resume               Continue an interrupted run-id, skipping completed arms"
    echo "  --skip-checks          Skip ruff+pytest (not recommended)"
    echo "  -h|--help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)        RUN_ID="$2"; shift 2 ;;
        --run-iteration) RUN_ITERATION="$2"; shift 2 ;;
        --branch)        BRANCH_OVERRIDE="$2"; shift 2 ;;
        --http-port)     HTTP_PORT="$2"; shift 2 ;;
        --llm-base-url)  LLM_BASE_URL="$2"; shift 2 ;;
        --llm-model)     LLM_MODEL="$2"; shift 2 ;;
        --resume)        RESUME=1; shift ;;
        --skip-checks)   SKIP_CHECKS=1; shift ;;
        -h|--help)       print_usage; exit 0 ;;
        *)               echo "Unknown argument: $1" >&2; print_usage; exit 1 ;;
    esac
done

# Resolve Run ID
if [[ -n "${RUN_ID}" && -n "${RUN_ITERATION}" ]]; then
    echo "error: Provide either --run-id OR --run-iteration, not both." >&2
    exit 1
fi

if [[ -z "${RUN_ID}" && -z "${RUN_ITERATION}" ]]; then
    echo "error: Either --run-id or --run-iteration is required" >&2
    print_usage
    exit 1
fi

if [[ -n "${RUN_ITERATION}" ]]; then
    if [[ ! "${RUN_ITERATION}" =~ ^[A-Za-z]$ ]]; then
        echo "error: --run-iteration must be a single letter (A-Z)" >&2
        exit 1
    fi

    echo "Auto-generating run-id..."
    APP_VERSION=$(uv run python -c "from hypotree import __version__; print(__version__)")
    
    # Sanitize model name for path safety (e.g. qwen3:8b -> qwen3-8b)
    SANITIZED_MODEL=$(echo "${LLM_MODEL}" | sed 's/[^A-Za-z0-9._-]/-/g')

    # Determine Git Branch
    BRANCH=""
    if [[ -n "${BRANCH_OVERRIDE}" ]]; then
        BRANCH="${BRANCH_OVERRIDE}"
    elif command -v git >/dev/null 2>&1; then
        BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    fi

    # Sanitize branch name for path safety
    SANITIZED_BRANCH=$(echo "${BRANCH}" | sed 's/[^A-Za-z0-9._-]/-/g')

    RUN_ID="v${APP_VERSION}_run-iteration~${RUN_ITERATION}_llm-model~${SANITIZED_MODEL}"
    if [[ -n "${SANITIZED_BRANCH}" ]]; then
        RUN_ID="${RUN_ID}_branch~${SANITIZED_BRANCH}"
    fi
    
    echo "Generated run-id: ${RUN_ID}"
fi

# Same constraint the Python side enforces: a run id is one safe path component.
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._~-]{1,128}$ ]]; then
    echo "error: run-id must be 1-128 chars of [A-Za-z0-9._~-]" >&2
    exit 1
fi

RUNS_DIR="eval/runs/${RUN_ID}"
WORKSPACE_ID="${RUN_ID}@hypotree-eval"

# Refuse to write into a directory that already holds results. The runner appends
# to its JSONL, so re-using a run id would interleave two runs into one log and
# silently double-count every metric downstream. --resume is the one exception:
# it re-enters the *same* run to finish arms that never completed, which is what
# you want after a crash 70 runs into a 90-run sweep rather than discarding the
# lot. Completed arms are skipped by their log, never re-run, so nothing doubles.
if compgen -G "${RUNS_DIR}/*.jsonl" >/dev/null && [[ "${RESUME}" -eq 0 ]]; then
    echo "error: ${RUNS_DIR} already contains logs — pick a fresh run-id (or pass --resume)" >&2
    exit 1
fi
mkdir -p "${RUNS_DIR}"

# The marker a finished arm leaves in its log. The logger's key is `event_type`;
# matching on `event` instead never fired, so --resume treated every completed
# arm as a crash, deleted its log and re-ran the whole sweep.
RUN_END_MARKER='"event_type": "run_end"'
# An episode the inference server killed writes run_end like any other, so the
# terminal record alone no longer proves an arm is done. It is only finished if
# it also did not end on an infrastructure fault — otherwise --resume would skip
# exactly the episodes that most need re-running.
INFRA_FAIL_MARKER='"infra_failed": true'

# Episodes that died on infrastructure, reported together at the end. A dropped
# connection at seed 10 of 30 used to abort the remaining 61 episodes; the sweep
# now carries on and tells you what to re-run with --resume.
FAILED_EPISODES=()

# True when a log holds a completed, non-infra-failed episode. Written so the
# function's exit status is the last command's, with no `&&` compound that
# `set -e` could read as a failure at the call site.
episode_complete() {
  local log="$1"
  grep -qs "${RUN_END_MARKER}" "${log}" || return 1
  ! grep -qs "${INFRA_FAIL_MARKER}" "${log}"
}


# Always clean up the landscape server, even on error or Ctrl-C. Without this an
# aborted run leaves a server bound to the port and the next seed scores its
# probes against the *previous* seed's landscape.
SRV=""
cleanup() {
    if [[ -n "${SRV}" ]] && kill -0 "${SRV}" 2>/dev/null; then
        kill "${SRV}" 2>/dev/null || true
        wait "${SRV}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 0. Sync + fast gate
uv sync
if [[ "${SKIP_CHECKS}" -eq 0 ]]; then
    uv run ruff check eval tests src
    uv run ruff format --check eval tests src
    uv run pytest -q
fi

# The frozen protocol is the single source of truth for seeds and arms.
SEEDS=$(uv run python -c "from eval.runner.config import TASK_SEEDS; print(' '.join(map(str, TASK_SEEDS)))")
ARMS=$(uv run python -c "from eval.runner.config import ALL_ARMS; print(' '.join(ALL_ARMS))")
N_SEEDS=$(echo "${SEEDS}" | wc -w)
N_ARMS=$(echo "${ARMS}" | wc -w)
echo "=== run-id=${RUN_ID}  workspace=${WORKSPACE_ID}  logs=${RUNS_DIR} ==="
echo "=== ${N_SEEDS} pre-registered seeds x ${N_ARMS} arms (${ARMS}) = $((N_SEEDS * N_ARMS)) runs ==="

# 1. Generate the held-out landscapes + briefings.
#    Deliberately NOT namespaced by run id: a landscape is a pure function of its
#    seed, so every run must face the identical task or runs are incomparable.
uv run python -m eval.environment.landscape_generator
uv run python -m eval.environment.fake_hypothesis_tree

# 2. Pre-flight — the analytic difficulty guarantee.
#    Every seed must require more probes than the first session allows, so no run
#    can be won before the first context reset. Checked BEFORE burning hours of
#    GPU time, not discovered afterwards from a diluted result.
uv run python - <<'PY'
import json, pathlib, sys
from eval.environment.landscape_scoring import min_reference_probes, reference_strategy_probes
from eval.runner.config import TASK_SEEDS

floor = min_reference_probes()
bad = []
for seed in TASK_SEEDS:
    data = json.loads(
        pathlib.Path(f"eval/environment/landscapes/landscape_seed_{seed}.json").read_text()
    )
    first_reset = data["session_breakpoints"][0]
    ref = reference_strategy_probes(seed)
    ok = ref > first_reset and first_reset < floor
    if not ok:
        bad.append(seed)
    print(
        f"  seed {seed}: reference={ref} probes, first reset at {first_reset}, "
        f"decoy={data['decoy_axis']}={data['decoy_value']} -> {'ok' if ok else 'FAIL'}"
    )
print(f"  global floor (any seed): {floor} probes")
if bad:
    sys.exit(f"difficulty calibration FAILED for seeds: {bad}")
print("Pre-flight OK: no seed is solvable before the first context reset.")
PY

# 2b. Pre-flight — the engine must be able to solve the task on its own.
#     Drives the engine with a scripted, perfectly-disciplined caller and asserts
#     the goal is actually reached on every seed. This separates "the engine's
#     search is broken" from "the model gave up", which look identical in the
#     logs and cost a day of reading JSONL to tell apart afterwards. A regression
#     in the substitution verdict once ended real episodes at 14, 16 and 23
#     probes out of 100 with the goal unmet, after several GPU-hours — this check
#     catches that class in two seconds, before any of them are spent.
uv run python -m eval.runner.engine_selfplay

# 3. Criterion 2 — TS-quality ablation (no LLM needed)
uv run python -m eval.runner.ablation_navigator eval/ --run-id "${RUN_ID}"

# 4. Criteria 1/3/4 — every arm on every seed.
#    One landscape server per seed, shared by all arms so they face an identical
#    environment.
for SEED in ${SEEDS}; do
  # A seed whose every arm already finished needs no server at all.
  if [[ "${RESUME}" -eq 1 ]]; then
    PENDING=0
    for ARM in ${ARMS}; do
      episode_complete "${RUNS_DIR}/seed-${SEED}-arm-${ARM}.jsonl" || PENDING=1
    done
    if [[ "${PENDING}" -eq 0 ]]; then
      echo "=== seed ${SEED}: all arms complete, skipping ==="
      continue
    fi
  fi

  uv run python eval/environment/landscape_server.py \
      "eval/environment/landscapes/landscape_seed_${SEED}.json" "${HTTP_PORT}" &
  SRV=$!

  # Wait for /health rather than sleeping a fixed second — a slow start would
  # otherwise silently fail every probe in the run.
  for _ in $(seq 1 30); do
      if curl -sf "http://localhost:${HTTP_PORT}/health" >/dev/null; then break; fi
      sleep 0.5
  done
  curl -sf "http://localhost:${HTTP_PORT}/health" || { echo "server failed to start" >&2; exit 1; }
  echo
  LANDSCAPE_URL="http://localhost:${HTTP_PORT}/evaluate"

  for ARM in ${ARMS}; do
    LOG="${RUNS_DIR}/seed-${SEED}-arm-${ARM}.jsonl"
    # A log carrying run_end is a finished arm. A log without one is a crash:
    # truncate it, because the runner appends and a half-written episode would
    # otherwise be concatenated with its retry into a single impossible history.
    if [[ "${RESUME}" -eq 1 && -f "${LOG}" ]]; then
      if episode_complete "${LOG}"; then
        echo "=== arm ${ARM} seed ${SEED}: complete, skipping ==="
        continue
      fi
      echo "=== arm ${ARM} seed ${SEED}: incomplete, re-running ==="
      rm -f "${LOG}"
    fi
    echo "=== Running arm ${ARM} for seed ${SEED} (run-id=${RUN_ID}) ==="
    # One episode must never take the sweep with it. `set -e` would abort the
    # whole run on a non-zero exit, which is how run I lost 61 of its 90
    # episodes to a single HTTP error; the failure is recorded and the sweep
    # moves on, to be picked up later with --resume.
    if ! uv run python -m eval.runner.runner eval/ "${SEED}" "${ARM}" \
        --run-id "${RUN_ID}" \
        --llm-backend openai \
        --llm-model "${LLM_MODEL}" \
        --llm-base-url "${LLM_BASE_URL}" \
        --landscape-url "${LANDSCAPE_URL}"; then
      echo "!!! arm ${ARM} seed ${SEED} FAILED — continuing with the next episode" >&2
      FAILED_EPISODES+=("seed ${SEED} arm ${ARM}")
    fi
  done

  cleanup
  SRV=""
done

# 4a. Report episodes that died outright, before any of the scoring runs.
#     They are excluded from the paired statistics rather than scored, so the
#     honest thing is to say so loudly here rather than let the arm counts
#     quietly disagree in the report.
if [[ "${#FAILED_EPISODES[@]}" -gt 0 ]]; then
  echo
  echo "!!! ${#FAILED_EPISODES[@]} episode(s) failed and are excluded from scoring:"
  printf '      %s\n' "${FAILED_EPISODES[@]}"
  echo "    Re-run just these with: $0 --run-iteration <SAME> --resume"
  echo
fi

# 4b. Surface censored episodes before anything is scored.
#     An episode that never met the goal is right-censored to the full budget and
#     scored as a maximum-cost loss, which is correct only when it genuinely
#     failed. One of run D's was a *baseline* arm killed by the harness after 25
#     productive probes with 75 unspent — the single largest value in the run,
#     biased toward the treatment, and visible only in §12 of a report generated
#     three steps later. Non-fatal, because a real failure is a real result, but
#     it belongs on the operator's screen while the run is still fresh.
uv run python - "${RUNS_DIR}" <<'PY'
import json, pathlib, sys

runs = pathlib.Path(sys.argv[1])
censored = []
for path in sorted(runs.glob("seed-*-arm-*.jsonl")):
    end = None
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event_type") == "run_end":
            end = ev
    if end is None:
        censored.append((path.name, "no run_end (crash)", "?"))
    elif not end.get("goals_met"):
        censored.append((path.name, str(end.get("reason")), end.get("step")))

if censored:
    print(f"  !! {len(censored)} episode(s) will be right-censored to the tool budget:")
    for name, reason, step in censored:
        print(f"     {name}: {reason} at step {step}")
    print("     Read the end reason before the gate: an episode that stopped with")
    print("     budget to spare is a defect, not a measurement of how hard the task is.")
else:
    print("  every episode reached the goal — no censoring in this run.")
PY

# 5. Score the frozen gate → emits GO / STOP / ITERATE
uv run python -m eval.analyse_gate eval/ --run-id "${RUN_ID}" | tee "${RUNS_DIR}/gate_decision.json"

# 6. Render the diagnostic report next to the raw logs.
uv run python -m eval.seed_reader --run-id "${RUN_ID}" --output "${RUNS_DIR}/REPORT.md"
echo "=== gate:   ${RUNS_DIR}/gate_decision.json ==="
echo "=== report: ${RUNS_DIR}/REPORT.md ==="