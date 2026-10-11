#!/usr/bin/env bash
# Watch where decode time goes, per process.
#
#   terminal 1:  nix run .#bench
#   terminal 2:  bash watch-decode.sh
#
# SCOPE: read from the model process and its children only. Host-wide counters are
# labelled. Disk is measured as the target's own major faults, never from the
# host's aggregate -- other activity on the box would be charged to the model.
#
#   majflt/s  The number that matters. Major faults = pages the process had to
#             actually read from block storage (the n-gram table is mmap'd, so
#             these never appear as read() traffic). Multiply by 4096 for bytes.
#             Near zero means the table is living in page cache.
#   huge%     Share of model memory in 2 MiB pages. The expert arena is ~58 GB
#             read scattered; at 4 KiB coverage nearly every read is a TLB miss.
#             Only relevant if you are CPU-bound rather than disk-bound.
#   MHz       Sustained clock. Large drops from 5600 mean AVX throttling.
#   cpu%      HOST-WIDE busy.
#
# PID=$(1) defaults to the first bench.py/serve.py it finds.

set -u

PID=${1:-$(pgrep -f 'bench\.py|serve\.py' | head -1)}
if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    echo "no bench.py/serve.py running -- start 'nix run .#bench' first" >&2
    exit 1
fi

# The CPU MoE workers are spawned children holding the expert arena, so the
# parent alone badly undercounts. Sum the whole family.
family() { echo "$PID"; pgrep -P "$PID" 2>/dev/null; }

sum_rollup() {                       # $1 = smaps_rollup field
    local t=0 v p
    for p in $(family); do
        v=$(awk -v k="^$1:" '$0 ~ k {print $2; exit}' "/proc/$p/smaps_rollup" 2>/dev/null)
        t=$((t + ${v:-0}))
    done; echo "$t"
}

# /proc/PID/stat fields 10 = minflt, 12 = majflt
sum_flt() {
    local t=0 v p
    for p in $(family); do
        v=$(awk '{print $12}' "/proc/$p/stat" 2>/dev/null)
        t=$((t + ${v:-0}))
    done; echo "$t"
}

mb() { awk -v b="${1:-0}" 'BEGIN{printf "%.1f", b/1048576}'; }

echo "watching pid $PID (and children)"
printf "%5s %8s %8s %6s %9s %9s %7s %6s\n" \
       t anonGB hugeGB "huge%" "majflt/s" "diskMB/s" MHz cpu%

maj0=$(sum_flt)
cpu0=$(awk '/^cpu /{print $2+$3+$4+$7+$8}' /proc/stat)   # user+nice+system+irq+softirq; idle is $6
t0=$(date +%s)

while kill -0 "$PID" 2>/dev/null; do
    anon=$(sum_rollup Anonymous);    anon=${anon:-0}
    huge=$(sum_rollup AnonHugePages); huge=${huge:-0}
    maj1=$(sum_flt)
    majrate=$(( maj1 - maj0 )); maj0=$maj1
    diskmb=$(( majrate * 4096 / 1048576 ))

    cpu1=$(awk '/^cpu /{print $2+$3+$4+$7+$8}' /proc/stat)
    pctcpu=0
    [ "${cpu1:-0}" -gt "${cpu0:-0}" ] && pctcpu=$(( (cpu1 - cpu0) * 100 / (cpu1 - cpu0) ))
    cpu0=$cpu1

    mhz=$(awk '/cpu MHz/{s+=$4; n++} END{if(n) printf "%.0f", s/n}' /proc/cpuinfo)
    [ "$anon" -gt 0 ] && hpct=$(( huge * 100 / anon )) || hpct=0
    printf "%4ds %8s %8s %5d%% %9d %9d %7s %5d%%\n" \
        "$(($(date +%s) - t0))" "$(mb "$anon")" "$(mb "$huge")" "$hpct" \
        "$majrate" "$diskmb" "${mhz:-?}" "$pctcpu"
    sleep 2
done

echo "pid $PID exited"