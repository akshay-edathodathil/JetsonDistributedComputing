#!/usr/bin/env bash
# cluster_setup.sh — Ray cluster control for GPU PC (head) + Jetson workers
# Usage: ./cluster_setup.sh [sync|setup|start|stop|status|restart]

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
HEAD_IP="192.168.1.1"
RAY_PORT=6379
DASHBOARD_PORT=8265
HEAD_VENV="/home/cams/Desktop/Akshay/.venv_mmpose"
WORKER_USER="nvidia"
WORKER_PROJECT_DIR="~/JetsonDistributedComp"
WORKER_VENV="~/JetsonDistributedComp/venv"
HEAD_OBJECT_STORE_GB=4
WORKER_OBJECT_STORE_GB=2
WORKER_NUM_CPUS=4

JETSON_IPS=(
    "192.168.1.2"   # j001
    "192.168.1.3"   # j002
    "192.168.1.4"   # j003
    "192.168.1.5"   # j004
)

# Password used only for the one-time SSH key deployment (setup_ssh_keys).
# After that, all connections use key auth and this is never read again.
WORKER_PASSWORD="nvidia"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

ssh_worker() {
    local ip="$1"; shift
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "${WORKER_USER}@${ip}" "$@"
}

# ── setup_ssh_keys: one-time passwordless SSH setup ───────────────────────────
cmd_setup_ssh_keys() {
    if ! command -v sshpass &>/dev/null; then
        err "sshpass not found. Install with: sudo apt install sshpass"
        exit 1
    fi
    if [[ ! -f ~/.ssh/id_ed25519.pub ]]; then
        log "No SSH key found. Generating ed25519 key..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
    fi
    log "Deploying SSH key to ${#JETSON_IPS[@]} Jetsons (password: ${WORKER_PASSWORD})..."
    local failed=0
    for ip in "${JETSON_IPS[@]}"; do
        log "  → $ip"
        if sshpass -p "${WORKER_PASSWORD}" ssh-copy-id \
            -o StrictHostKeyChecking=no \
            -i ~/.ssh/id_ed25519.pub \
            "${WORKER_USER}@${ip}" 2>/dev/null; then
            log "  ✓ $ip — key installed"
        else
            err "  ✗ $ip — failed (wrong password or unreachable?)"
            (( failed++ )) || true
        fi
    done
    [[ $failed -eq 0 ]] && log "SSH key setup complete. Password no longer needed." \
                        || err "$failed Jetson(s) failed."
    return $failed
}

# ── sync: push project code to all Jetsons ────────────────────────────────────
cmd_sync() {
    log "Syncing code to ${#JETSON_IPS[@]} Jetsons..."
    local failed=0
    for ip in "${JETSON_IPS[@]}"; do
        log "  → $ip"
        if rsync -avz --progress \
            --exclude 'venv/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --exclude 'results/' \
            --exclude '.git/' \
            "${SCRIPT_DIR}/" \
            "${WORKER_USER}@${ip}:${WORKER_PROJECT_DIR}/"; then
            log "  ✓ $ip synced"
        else
            err "  ✗ $ip sync failed"
            (( failed++ )) || true
        fi
    done
    [[ $failed -eq 0 ]] && log "Sync complete." || err "$failed Jetson(s) failed to sync."
    return $failed
}

# ── setup: create venv on each Jetson ─────────────────────────────────────────
cmd_setup() {
    log "Setting up worker environments on ${#JETSON_IPS[@]} Jetsons..."
    # First sync code so setup_worker_env.sh is present
    cmd_sync

    local failed=0
    for ip in "${JETSON_IPS[@]}"; do
        log "  → $ip: running setup_worker_env.sh"
        if ssh_worker "$ip" "bash ${WORKER_PROJECT_DIR}/setup_worker_env.sh"; then
            log "  ✓ $ip setup complete"
        else
            err "  ✗ $ip setup failed"
            (( failed++ )) || true
        fi
    done
    [[ $failed -eq 0 ]] && log "Worker setup complete." || err "$failed Jetson(s) failed setup."
    return $failed
}

# ── start_head: launch Ray head node on this machine ─────────────────────────
cmd_start_head() {
    log "Starting Ray head node on ${HEAD_IP}:${RAY_PORT}..."
    # shellcheck disable=SC1090
    source "${HEAD_VENV}/bin/activate"

    if ray status &>/dev/null; then
        log "Ray already running on head. Skipping."
        return 0
    fi

    ray start \
        --head \
        --port="${RAY_PORT}" \
        --dashboard-host="0.0.0.0" \
        --dashboard-port="${DASHBOARD_PORT}" \
        --num-cpus="$(nproc)" \
        --num-gpus=1 \
        --object-store-memory="$(( HEAD_OBJECT_STORE_GB * 1024 * 1024 * 1024 ))"

    log "Ray head started. Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"
}

# ── start_workers: SSH each Jetson and start Ray worker ───────────────────────
cmd_start_workers() {
    local num_workers="${1:-${#JETSON_IPS[@]}}"
    log "Starting Ray workers on ${num_workers} Jetson(s)..."

    local failed=0
    for i in $(seq 0 $(( num_workers - 1 ))); do
        local ip="${JETSON_IPS[$i]}"
        log "  → $ip"
        if ssh_worker "$ip" bash <<EOF
set -e
source ${WORKER_VENV}/bin/activate
# Stop any existing Ray instance on this worker
ray stop --force 2>/dev/null || true
sleep 1
ray start \\
    --address="${HEAD_IP}:${RAY_PORT}" \\
    --num-gpus=1 \\
    --num-cpus=${WORKER_NUM_CPUS} \\
    --object-store-memory=$(( WORKER_OBJECT_STORE_GB * 1024 * 1024 * 1024 )) \\
    --resources='{"jetson": 1}' \\
    --node-ip-address="${ip}"
echo "Ray worker started on ${ip}"
EOF
        then
            log "  ✓ $ip worker started"
        else
            err "  ✗ $ip worker failed to start"
            (( failed++ )) || true
        fi
    done
    [[ $failed -eq 0 ]] && log "All workers started." || err "$failed worker(s) failed."
    return $failed
}

# ── start: head + all workers ─────────────────────────────────────────────────
cmd_start() {
    local num_workers="${1:-${#JETSON_IPS[@]}}"
    cmd_start_head
    cmd_start_workers "$num_workers"
    log "Cluster running. Workers: $num_workers / ${#JETSON_IPS[@]}"
}

# ── stop: shut down Ray on all nodes ─────────────────────────────────────────
cmd_stop() {
    log "Stopping Ray on all nodes..."

    for ip in "${JETSON_IPS[@]}"; do
        log "  → $ip: stopping worker"
        ssh_worker "$ip" "source ${WORKER_VENV}/bin/activate && ray stop --force" 2>/dev/null || true
    done

    log "  → head: stopping"
    # shellcheck disable=SC1090
    source "${HEAD_VENV}/bin/activate"
    ray stop --force 2>/dev/null || true
    log "All nodes stopped."
}

# ── status: show cluster state ────────────────────────────────────────────────
cmd_status() {
    # shellcheck disable=SC1090
    source "${HEAD_VENV}/bin/activate"
    echo ""
    echo "═══ Ray Cluster Status ═══"
    ray status 2>/dev/null || echo "Ray is not running on head."
    echo ""
    echo "═══ Jetson Connectivity ═══"
    for ip in "${JETSON_IPS[@]}"; do
        if ssh_worker "$ip" "echo ok" &>/dev/null; then
            local ray_running
            ray_running=$(ssh_worker "$ip" \
                "source ${WORKER_VENV}/bin/activate && ray status 2>/dev/null && echo RUNNING || echo STOPPED" 2>/dev/null)
            echo "  $ip — SSH: OK | Ray: $ray_running"
        else
            echo "  $ip — SSH: UNREACHABLE"
        fi
    done
}

# ── restart: stop then start ──────────────────────────────────────────────────
cmd_restart() {
    local num_workers="${1:-${#JETSON_IPS[@]}}"
    cmd_stop
    sleep 2
    cmd_start "$num_workers"
}

# ── sync-models: push converted_models/ to all Jetsons ───────────────────────
cmd_sync_models() {
    log "Syncing converted_models/ to ${#JETSON_IPS[@]} Jetsons..."
    local failed=0
    for ip in "${JETSON_IPS[@]}"; do
        log "  → $ip"
        if rsync -avz --progress --checksum \
            --exclude '*.engine' \
            "${SCRIPT_DIR}/converted_models/" \
            "${WORKER_USER}@${ip}:${WORKER_PROJECT_DIR}/converted_models/"; then
            log "  ✓ $ip models synced"
        else
            err "  ✗ $ip model sync failed"
            (( failed++ )) || true
        fi
    done
    [[ $failed -eq 0 ]] && log "Models sync complete." || err "$failed Jetson(s) failed."
    return $failed
}

# ── run-session: run the inference pipeline on all Jetsons ───────────────────
cmd_run_session() {
    log "Running session pipeline via Ansible..."
    if ! command -v ansible-playbook &>/dev/null; then
        err "ansible-playbook not found."
        exit 1
    fi
    ansible-playbook -i "${SCRIPT_DIR}/inventory.yaml" "${SCRIPT_DIR}/run_session.yaml" "$@"
}

# ── Entrypoint ────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 <command> [options]

Cluster commands:
  ssh-keys          One-time: deploy SSH key to all Jetsons (needs password once)
  sync              Rsync project code to all Jetsons
  start [N]         Start Ray head + N workers (default: all 4)
  start-head        Start Ray head node only
  start-workers [N] Start N Ray worker nodes only
  stop              Stop Ray on all nodes
  restart [N]       Stop then start with N workers
  status            Show cluster status

Session / inference commands:
  sync-models       Rsync converted_models/ to all Jetsons (skips .engine files)
  run-session       Run per-video inference pipeline on all Jetsons via Ansible
                    (edit video_assignments in run_session.yaml first)

First-time setup order:
  $0 ssh-keys       # Deploy SSH keys (once only)
  $0 sync           # Sync code to Jetsons
  $0 sync-models    # Sync model files to Jetsons

Per-session inference:
  $0 run-session    # Distribute videos, run pipeline, collect CSVs

Examples:
  $0 sync-models
  $0 run-session
  $0 run-session --limit j001   # Test on one Jetson only
EOF
}

case "${1:-}" in
    ssh-keys)       cmd_setup_ssh_keys ;;
    sync)           cmd_sync ;;
    setup)          cmd_setup ;;
    start)          cmd_start "${2:-}" ;;
    start-head)     cmd_start_head ;;
    start-workers)  cmd_start_workers "${2:-}" ;;
    stop)           cmd_stop ;;
    restart)        cmd_restart "${2:-}" ;;
    status)         cmd_status ;;
    sync-models)    cmd_sync_models ;;
    run-session)    shift; cmd_run_session "$@" ;;
    *)              usage; exit 1 ;;
esac
