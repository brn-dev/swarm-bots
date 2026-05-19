#!/usr/bin/env sh

set -eu

usage() {
    cat <<'EOF'
Usage: wait_then_run_repeated.sh REQUIRED_FREE_VRAM SCRIPT_PATH [RUNS] [OPTIONS] [-- SCRIPT_ARGS...]

Wait until the selected GPU has enough free VRAM for a stable period, then call
scripts/run_repeated.sh.

Arguments:
  REQUIRED_FREE_VRAM  Required free VRAM. Plain numbers are MiB. Supported
                      suffixes: M, MiB, G, GiB, T, TiB. Examples: 24000, 24G.
  SCRIPT_PATH         Training script path passed to run_repeated.sh.
  RUNS                Number of runs passed to run_repeated.sh. Defaults to 5.

Options:
  --gpu INDEX             GPU index for nvidia-smi. Defaults to 0.
  --stable-for SECONDS    Required continuous availability. Defaults to 300.
  --check-interval SECONDS
                          Seconds between VRAM checks. Defaults to 30.
  --python PYTHON         Python executable passed to run_repeated.sh.
                          Defaults to "python".
  --delay SECONDS         Delay between repeated runs. Defaults to 0.
  --stop-file PATH        Stop-request file passed to run_repeated.sh.
  --lock-dir PATH         Lock directory for serializing the VRAM wait/start
                          decision. Defaults to a per-GPU directory under
                          ${TMPDIR:-/tmp}.
  --no-lock               Disable wait/start locking.
  --help                  Show this help text.

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

is_positive_integer() {
    is_non_negative_integer "$1" && [ "$1" -gt 0 ]
}

parse_memory_mib() {
    memory_text=$1

    case "$memory_text" in
        *[Tt][Ii][Bb])
            memory_number=${memory_text%[Tt][Ii][Bb]}
            memory_multiplier=1048576
            ;;
        *[Tt][Bb])
            memory_number=${memory_text%[Tt][Bb]}
            memory_multiplier=1048576
            ;;
        *[Tt])
            memory_number=${memory_text%[Tt]}
            memory_multiplier=1048576
            ;;
        *[Gg][Ii][Bb])
            memory_number=${memory_text%[Gg][Ii][Bb]}
            memory_multiplier=1024
            ;;
        *[Gg][Bb])
            memory_number=${memory_text%[Gg][Bb]}
            memory_multiplier=1024
            ;;
        *[Gg])
            memory_number=${memory_text%[Gg]}
            memory_multiplier=1024
            ;;
        *[Mm][Ii][Bb])
            memory_number=${memory_text%[Mm][Ii][Bb]}
            memory_multiplier=1
            ;;
        *[Mm][Bb])
            memory_number=${memory_text%[Mm][Bb]}
            memory_multiplier=1
            ;;
        *[Mm])
            memory_number=${memory_text%[Mm]}
            memory_multiplier=1
            ;;
        *)
            memory_number=$memory_text
            memory_multiplier=1
            ;;
    esac

    if ! is_positive_integer "$memory_number"; then
        printf 'Invalid VRAM amount: %s\n' "$memory_text" >&2
        exit 1
    fi

    printf '%s\n' "$((memory_number * memory_multiplier))"
}

query_free_vram_mib() {
    query_gpu_index=$1
    free_vram=$(
        nvidia-smi \
            --id="$query_gpu_index" \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null \
            | tr -d '[:space:]'
    )

    if ! is_non_negative_integer "$free_vram"; then
        printf 'Could not read free VRAM for GPU %s with nvidia-smi.\n' "$query_gpu_index" >&2
        exit 1
    fi

    printf '%s\n' "$free_vram"
}

cleanup_lock() {
    if [ "${lock_acquired:-0}" -eq 1 ]; then
        rm -f -- "$lock_dir/pid"
        rmdir -- "$lock_dir" 2>/dev/null || true
        lock_acquired=0
    fi
}

acquire_lock() {
    while ! mkdir -- "$lock_dir" 2>/dev/null; do
        lock_pid=
        if [ -f "$lock_dir/pid" ]; then
            lock_pid=$(cat -- "$lock_dir/pid" 2>/dev/null || true)
        fi

        if is_positive_integer "${lock_pid:-}" && ! kill -0 "$lock_pid" 2>/dev/null; then
            printf 'Removing stale lock: %s\n' "$lock_dir"
            rm -f -- "$lock_dir/pid"
            rmdir -- "$lock_dir" 2>/dev/null || true
            continue
        fi

        if [ -n "${lock_pid:-}" ]; then
            printf 'Waiting for GPU %s lock held by PID %s: %s\n' "$gpu_index" "$lock_pid" "$lock_dir"
        else
            printf 'Waiting for GPU %s lock: %s\n' "$gpu_index" "$lock_dir"
        fi
        sleep "$check_interval_seconds"
    done

    lock_acquired=1
    printf '%s\n' "$$" > "$lock_dir/pid"
}

if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

required_free_vram_mib=$(parse_memory_mib "$1")
shift

if [ $# -eq 0 ]; then
    printf 'Missing SCRIPT_PATH.\n' >&2
    usage >&2
    exit 1
fi

script_path=$1
shift

runs=5
if [ $# -gt 0 ] && is_non_negative_integer "$1"; then
    runs=$1
    shift
fi

gpu_index=0
stable_for_seconds=300
check_interval_seconds=30
python_executable=python
delay_seconds=0
stop_file=
use_lock=1
lock_dir=

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --gpu\n' >&2
                exit 1
            fi
            if ! is_non_negative_integer "$2"; then
                printf 'Invalid GPU index: %s\n' "$2" >&2
                exit 1
            fi
            gpu_index=$2
            shift 2
            ;;
        --stable-for)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --stable-for\n' >&2
                exit 1
            fi
            if ! is_non_negative_integer "$2"; then
                printf 'Invalid stable duration: %s\n' "$2" >&2
                exit 1
            fi
            stable_for_seconds=$2
            shift 2
            ;;
        --check-interval)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --check-interval\n' >&2
                exit 1
            fi
            if ! is_positive_integer "$2"; then
                printf 'Invalid check interval: %s\n' "$2" >&2
                exit 1
            fi
            check_interval_seconds=$2
            shift 2
            ;;
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
        --stop-file)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --stop-file\n' >&2
                exit 1
            fi
            stop_file=$2
            shift 2
            ;;
        --lock-dir)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --lock-dir\n' >&2
                exit 1
            fi
            lock_dir=$2
            shift 2
            ;;
        --no-lock)
            use_lock=0
            shift
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
run_repeated_script=$script_dir/run_repeated.sh

if [ ! -f "$run_repeated_script" ]; then
    printf 'run_repeated.sh not found next to this script: %s\n' "$run_repeated_script" >&2
    exit 1
fi

if [ -z "$lock_dir" ]; then
    lock_dir=${TMPDIR:-/tmp}/swarmbots_wait_then_run_repeated_gpu_${gpu_index}.lock
fi

lock_acquired=0
if [ "$use_lock" -ne 0 ]; then
    trap cleanup_lock EXIT
    trap 'cleanup_lock; exit 130' INT
    trap 'cleanup_lock; exit 143' TERM
    acquire_lock
fi

stable_started_at=

printf 'Waiting for GPU %s to have at least %s MiB free for %s seconds.\n' \
    "$gpu_index" "$required_free_vram_mib" "$stable_for_seconds"

while :; do
    now=$(date '+%s')
    free_vram_mib=$(query_free_vram_mib "$gpu_index")

    if [ "$free_vram_mib" -ge "$required_free_vram_mib" ]; then
        if [ -z "$stable_started_at" ]; then
            stable_started_at=$now
            printf 'GPU %s free VRAM is now %s MiB; starting stable timer.\n' "$gpu_index" "$free_vram_mib"
        fi

        stable_elapsed=$((now - stable_started_at))
        if [ "$stable_elapsed" -ge "$stable_for_seconds" ]; then
            printf 'GPU %s stayed above %s MiB for %s seconds. Starting repeated run.\n' \
                "$gpu_index" "$required_free_vram_mib" "$stable_elapsed"
            break
        fi
    else
        if [ -n "$stable_started_at" ]; then
            printf 'GPU %s free VRAM dropped to %s MiB; resetting stable timer.\n' "$gpu_index" "$free_vram_mib"
        else
            printf 'GPU %s free VRAM is %s MiB; waiting for %s MiB.\n' \
                "$gpu_index" "$free_vram_mib" "$required_free_vram_mib"
        fi
        stable_started_at=
    fi

    sleep "$check_interval_seconds"
done

cleanup_lock

if [ -n "$stop_file" ]; then
    sh "$run_repeated_script" "$script_path" "$runs" \
        --python "$python_executable" \
        --delay "$delay_seconds" \
        --stop-file "$stop_file" \
        -- "$@"
else
    sh "$run_repeated_script" "$script_path" "$runs" \
        --python "$python_executable" \
        --delay "$delay_seconds" \
        -- "$@"
fi
