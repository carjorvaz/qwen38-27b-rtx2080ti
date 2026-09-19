#!/usr/bin/env bash
set -euo pipefail

# Apply the full stack to a pristine vLLM 0.28.0 checkout, in the order it is
# installed: patches/ (the 3090 series, glob order, dflash2-backport skipped
# because DFlash2 is native in 0.28.0), then patches-turing/ in `series` order.
# GNU patch is the tool that installs this stack, so GNU patch defines "applies".
#
# usage: bash patches-turing/check_vllm_series.sh /path/to/vllm-v0.28.0/vllm

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE=${1:?usage: bash patches-turing/check_vllm_series.sh /path/to/vllm-v0.28.0/vllm}
VLLM_SOURCE=$(cd -- "$VLLM_SOURCE" && pwd)
GIT_ROOT=$(git -C "$VLLM_SOURCE" rev-parse --show-toplevel)
[ -f "$VLLM_SOURCE/__init__.py" ] || { echo "not a vllm package directory: $VLLM_SOURCE" >&2; exit 1; }

echo "== pass 1: the 3090 series, GNU patch, glob order"
git -C "$GIT_ROOT" checkout -q -- . && git -C "$GIT_ROOT" clean -qfd
count=0; offsets=0
for p in "$HERE"/patches/*.patch; do
  name=$(basename "$p")
  [ "$name" = "dflash2-backport.patch" ] && { echo "   skip $name (native in 0.28.0)"; continue; }
  out=$(patch -p1 --forward --no-backup-if-mismatch -d "$VLLM_SOURCE" < "$p" 2>&1) || {
    echo "FAILED: $name"; echo "$out" | sed 's/^/    /'; exit 1
  }
  n=$(printf '%s\n' "$out" | grep -c "offset\|with fuzz" || true)
  [ "$n" -gt 0 ] && { echo "   $name (applied, $n hunk(s) with offset)"; offsets=$((offsets+1)); }
  count=$((count+1))
done
git -C "$GIT_ROOT" diff --quiet && { echo "ERROR: the series applied but changed nothing" >&2; exit 1; }
echo "   $count patches applied, $offsets with an offset"

echo "== pass 2: the Turing series, GNU patch, series order"
count=0; offsets=0
while read -r name; do
  [ -n "$name" ] || continue
  case "$name" in \#*) continue ;; esac
  p="$HERE/patches-turing/$name"
  [ -f "$p" ] || { echo "FAILED: patches-turing/$name is missing" >&2; exit 1; }
  out=$(patch -p1 --forward --no-backup-if-mismatch -d "$VLLM_SOURCE" < "$p" 2>&1) || {
    echo "FAILED: $name"; echo "$out" | sed 's/^/    /'; exit 1
  }
  n=$(printf '%s\n' "$out" | grep -c "offset\|with fuzz" || true)
  [ "$n" -gt 0 ] && { echo "   $name (applied, $n hunk(s) with offset)"; offsets=$((offsets+1)); }
  count=$((count+1))
done < "$HERE/patches-turing/series"
echo "   $count patches applied, $offsets with an offset"

echo "OK: the full series applies to vLLM 0.28.0"
