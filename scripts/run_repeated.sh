#!/usr/bin/env sh

set -eu

usage() {
    cat <<'EOF'
Usage: run_repeated.sh SCRIPT_PATH [RUNS] [--python PYTHON] [--delay SECONDS] [-- SCRIPT_ARGS...]

Arguments:
  SCRIPT_PATH         Training script path. Relative paths are resolved against:
                      current directory, repo root, and the scripts directory.
  RUNS                Number of runs. Defaults to 5.

Options:
  --python PYTHON     Python executable to use. Defaults to "python".
  --delay SECONDS     Delay between runs. Defaults to 0.
  --help              Show this help text.

Any arguments after `--` are forwarded to the training script.
EOF
}

is_non_negative_integer() {
    case "$1" in
        ''|*[!0-9]*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

abspath() {
    target_path=$1
    target_dir=$(dirname -- "$target_path")
    target_name=$(basename -- "$target_path")
    (
        cd -- "$target_dir"
        printf '%s/%s\n' "$(pwd)" "$target_name"
    )
}

resolve_training_script_path() {
    requested_path=$1
    shift

    case "$requested_path" in
        /*|[A-Za-z]:/*)
            if [ ! -f "$requested_path" ]; then
                printf 'Training script not found: %s\n' "$requested_path" >&2
                exit 1
            fi
            abspath "$requested_path"
            return 0
            ;;
    esac

    for base_directory in "$@"; do
        candidate_path=$base_directory/$requested_path
        if [ -f "$candidate_path" ]; then
            abspath "$candidate_path"
            return 0
        fi
    done

    printf 'Training script not found: %s\n' "$requested_path" >&2
    exit 1
}

if [ $# -eq 0 ]; then
    usage >&2
    exit 1
fi

script_path=
runs=5
python_executable=python
delay_seconds=0

if [ "$1" = "--help" ]; then
    usage
    exit 0
fi

script_path=$1
shift

if [ $# -gt 0 ] && is_non_negative_integer "$1"; then
    runs=$1
    shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --python)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --python\n' >&2
                exit 1
            fi
            python_executable=$2
            shift 2
            ;;
        --delay)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --delay\n' >&2
                exit 1
            fi
            if ! is_non_negative_integer "$2"; then
                printf 'Invalid delay value: %s\n' "$2" >&2
                exit 1
            fi
            delay_seconds=$2
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if ! is_non_negative_integer "$runs"; then
    printf 'Invalid run count: %s\n' "$runs" >&2
    exit 1
fi

script_dir=$(
    cd -- "$(dirname -- "$0")"
    pwd
)
repo_root=$(
    cd -- "$script_dir/.."
    pwd
)
current_dir=$(pwd)
training_script=$(resolve_training_script_path "$script_path" "$current_dir" "$repo_root" "$script_dir")
original_pythonpath=${PYTHONPATH-}
failed_runs=
has_failures=0
run_index=1

case "$(uname -s 2>/dev/null || printf unknown)" in
    CYGWIN*|MINGW*|MSYS*)
        pythonpath_separator=';'
        ;;
    *)
        pythonpath_separator=':'
        ;;
esac

while [ "$run_index" -le "$runs" ]; do
    started_at=$(date '+%Y-%m-%dT%H:%M:%S')
    printf '[%s/%s] Starting %s at %s\n' "$run_index" "$runs" "$training_script" "$started_at"

    if [ -n "$original_pythonpath" ]; then
        pythonpath_value=$repo_root$pythonpath_separator$original_pythonpath
    else
        pythonpath_value=$repo_root
    fi

    if (
        export PYTHONPATH=$pythonpath_value
        cd -- "$script_dir"
        "$python_executable" "$training_script" "$@"
    ); then
        exit_code=0
    else
        exit_code=$?
    fi

    finished_at=$(date '+%Y-%m-%dT%H:%M:%S')
    if [ "$exit_code" -eq 0 ]; then
        printf '[%s/%s] Finished successfully at %s\n' "$run_index" "$runs" "$finished_at"
    else
        has_failures=1
        failed_runs="${failed_runs}${run_index}\t${exit_code}\t${started_at}\t${finished_at}
"
        printf '[%s/%s] Failed with exit code %s at %s\n' "$run_index" "$runs" "$exit_code" "$finished_at" >&2
    fi

    if [ "$run_index" -lt "$runs" ] && [ "$delay_seconds" -gt 0 ]; then
        sleep "$delay_seconds"
    fi

    run_index=$((run_index + 1))
done

if [ "$has_failures" -ne 0 ]; then
    printf '\nFailed runs:\n'
    printf 'Run\tExitCode\tStartedAt\tFinishedAt\n'
    printf '%s' "$failed_runs"
    exit 1
fi

printf '\nAll %s runs finished successfully.\n' "$runs"
exit 0
