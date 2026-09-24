#!/usr/bin/env bash
set -euo pipefail

# Validate the full patch stack against a pristine checkout of the pinned vLLM
# (the pin is docker/requirements.txt; this is what CI checks out and what the
# Dockerfile installs). Two things are checked:
#
#   1. upstream's own checker (patches/check_vllm_series.sh), which validates
#      the 3090 series in patches/series order and the five contractual DFlash
#      patches with `git apply`.
#   2. the whole stack as it installs: patches/series (skipping
#      dflash2-backport.patch, native since 0.28) and then patches-turing/series,
#      both with GNU patch at --fuzz 0. GNU patch is the tool that installs this
#      stack, so GNU patch defines "applies"; --fuzz 0 fails a patch whose hunk
#      context no longer exists instead of letting it land by guess.
#
# Both series files and their directories must agree exactly: a patch absent
# from a series file is never applied by the build, and one listed but missing
# is a typo.
#
# usage: bash patches-turing/check_vllm_series.sh /path/to/vllm-v0.29.0/vllm
#        (the package directory inside the checkout, not its root)

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE=${1:?usage: bash patches-turing/check_vllm_series.sh /path/to/vllm-<pinned tag>/vllm}
# -P: resolve symlinks before comparing paths (on macOS /tmp is /private/tmp,
# and a mismatched prefix makes the reject paths wrong instead of failing loudly).
VLLM_SOURCE=$(cd -P -- "$VLLM_SOURCE" && pwd)
GIT_ROOT=$(git -C "$VLLM_SOURCE" rev-parse --show-toplevel)
GIT_ROOT=$(cd -P -- "$GIT_ROOT" && pwd)
[ -f "$VLLM_SOURCE/__init__.py" ] || { echo "not a vllm package directory: $VLLM_SOURCE" >&2; exit 1; }

read_series() {  # strip comments and blank lines, print one name per line
  sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$1"
}

check_series_dir() {  # dir, series file, label
  local dir=$1 series=$2 label=$3 on_disk in_series
  on_disk=$(for f in "$dir"/*.patch; do basename "$f"; done | sort)
  in_series=$(read_series "$series" | sort)
  [ "$on_disk" = "$in_series" ] || {
    echo "ERROR: the $label series and its directory disagree:" >&2
    LC_ALL=C comm -3 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$in_series") | sed 's/^/    /' >&2
    exit 1
  }
}

apply_series() {  # dir, series file, label, [names to skip...]
  local dir=$1 series=$2 label=$3; shift 3
  local -a skip=("$@")
  local name p out n count=0 offsets=0 s hit
  while IFS= read -r name; do
    hit=0
    for s in "${skip[@]}"; do
      [ -n "$s" ] && [ "$name" = "$s" ] && hit=1 && break
    done
    [ "$hit" = 1 ] && { echo "   skip $name"; continue; }
    p="$dir/$name"
    [ -f "$p" ] || { echo "FAILED: $label/$name is listed but missing" >&2; exit 1; }
    out=$(patch -p1 --forward --no-backup-if-mismatch --fuzz 0 -d "$VLLM_SOURCE" < "$p" 2>&1) || {
      echo "FAILED: $label/$name (a hunk's context does not exist in this tree; regenerate the patch against the pin)" >&2
      echo "$out" | sed 's/^/    /' >&2
      exit 1
    }
    n=$(printf '%s\n' "$out" | grep -c "offset" || true)
    [ "$n" -gt 0 ] && { echo "   $name (applied, $n hunk(s) at an offset, context exact)"; offsets=$((offsets+1)); }
    count=$((count+1))
  done < <(read_series "$series")
  echo "   $label: $count patches applied with exact context, $offsets at an offset, 0 with fuzz"
}

check_series_dir "$HERE/patches" "$HERE/patches/series" "upstream 3090"
check_series_dir "$HERE/patches-turing" "$HERE/patches-turing/series" "Turing"

if [ -f "$HERE/patches/check_vllm_series.sh" ]; then
  echo "== upstream series, upstream checker (GNU patch pass + git apply pass)"
  bash "$HERE/patches/check_vllm_series.sh" "$VLLM_SOURCE" | sed 's/^/   /'
fi

echo "== pass 1: upstream 3090 series, GNU patch, patches/series order"
git -C "$GIT_ROOT" checkout -q -- . && git -C "$GIT_ROOT" clean -qfd
apply_series "$HERE/patches" "$HERE/patches/series" "upstream 3090" dflash2-backport.patch
git -C "$GIT_ROOT" diff --quiet && { echo "ERROR: the series applied but changed nothing" >&2; exit 1; }

echo "== pass 2: Turing series, GNU patch, patches-turing/series order"
apply_series "$HERE/patches-turing" "$HERE/patches-turing/series" "Turing"

git -C "$GIT_ROOT" diff --check
echo "patch integrity: OK (upstream 3090 series + Turing series, fuzz 0)"
