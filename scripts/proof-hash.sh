#!/usr/bin/env bash
# Usage: ./scripts/proof-hash.sh [--base=REV] [--rings=LIST] [--sets=LIST] [--runs=N] [--dir=PATH]
# Check that the working tree produces byte-identical proofs to a baseline revision.
#
# The binary writes its wire proof to $ROKOKO_PROOF_OUT once the proof is complete
# (scripts/proof-hash.patch, in src/protocol/parties/executor.rs). The baseline is checked out in
# a temporary git worktree with that same patch applied and nothing else; the working tree must
# already contain it. Each ring and checkout gets its own --target-dir under --dir, kept between
# runs. Every set runs --runs times per checkout, so a nondeterministic proof shows up as a
# mismatch within one checkout.
#
# Defaults: --base = merge-base of HEAD and origin/main, --rings=rings/default.toml,rings/n128_d4.toml,
# --sets=p-26,p-28, --runs=2, --dir=${TMPDIR:-/tmp}/rokoko-proof-hash.
# Exit status: 0 if every hash matches, 1 if one differs, 2 on a usage or build error.

set -euo pipefail

die() { echo "$*" >&2; exit 2; }

repo=$(git rev-parse --show-toplevel)
patch=$repo/scripts/proof-hash.patch
base='' rings='rings/default.toml,rings/n128_d4.toml' sets='p-26,p-28' runs=2
dir=${TMPDIR:-/tmp}/rokoko-proof-hash
for arg; do
  case $arg in
    --base=*) base=${arg#*=} ;;
    --rings=*) rings=${arg#*=} ;;
    --sets=*) sets=${arg#*=} ;;
    --runs=*) runs=${arg#*=} ;;
    --dir=*) dir=${arg#*=} ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done
[[ $runs =~ ^[1-9][0-9]*$ ]] || die "--runs must be a positive integer"
base=$(git -C "$repo" rev-parse --verify "${base:-$(git -C "$repo" merge-base HEAD origin/main)}^{commit}")

if command -v sha256sum >/dev/null; then sha() { sha256sum "$1" | cut -d' ' -f1; }
else sha() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi

git -C "$repo" apply --reverse --check "$patch" 2>/dev/null \
  || die "the working tree lacks scripts/proof-hash.patch; apply it with git apply"

mkdir -p "$dir"
dir=$(cd "$dir" && pwd)
worktree=$dir/baseline
git -C "$repo" worktree remove --force "$worktree" 2>/dev/null || rm -rf "$worktree"
git -C "$repo" worktree prune
cleanup() { git -C "$repo" worktree remove --force "$worktree" 2>/dev/null || true; }
trap cleanup EXIT
git -C "$repo" worktree add --quiet --detach "$worktree" "$base"
# A baseline that already carries the hook (the patch was merged) is used as is.
if ! git -C "$worktree" apply --reverse --check "$patch" 2>/dev/null; then
  git -C "$worktree" apply "$patch"
fi
echo "baseline: $(git -C "$repo" log -1 --format='%h %s' "$base") (+ proof-hash.patch)"
echo "working tree: $(git -C "$repo" log -1 --format='%h %s' HEAD) (+ uncommitted changes)"
echo "proof-hash.patch only touches: $(git -C "$worktree" diff --name-only | tr '\n' ' ')"

status=0
results=$dir/results.txt
: > "$results"
for ring in ${rings//,/ }; do
  stem=$(basename "$ring" .toml)
  for side in baseline head; do
    if [ "$side" = baseline ]; then src=$worktree; else src=$repo; fi
    [ -f "$src/$ring" ] || die "$side has no $ring"
    target=$dir/target-$side-$stem
    echo "building $side for $ring"
    (cd "$src" && ROKOKO_RING=$ring cargo +nightly build --release --target-dir "$target") \
      > "$dir/build-$side-$stem.log" 2>&1 || die "the $side build failed for $ring; see $dir/build-$side-$stem.log"
    for set in ${sets//,/ }; do
      for run in $(seq "$runs"); do
        out=$dir/$side-$stem-$set-$run
        start=$(date +%s)
        (cd "$src" && ROKOKO_PROOF_OUT=$out.bin "$target/release/rokoko" "$set" > "$out.log" 2>&1) \
          || die "$side $ring $set run $run failed; see $out.log"
        seconds=$(( $(date +%s) - start ))
        size=$(grep -o 'Wire proof size: [0-9.]* KB' "$out.log" | grep -o '[0-9.]* KB' || true)
        echo "$stem $set $side $run $(sha "$out.bin") $(wc -c < "$out.bin" | tr -d ' ') ${seconds}s $size" \
          | tee -a "$results"
      done
    done
  done
done

echo
printf '%-10s %-5s %-17s %-17s %-6s %s\n' ring set 'baseline sha256' 'head sha256' match runs
for ring in ${rings//,/ }; do
  stem=$(basename "$ring" .toml)
  for set in ${sets//,/ }; do
    # One hash per side when its runs agree, else "differs between runs".
    side_hash() {
      awk -v r="$stem" -v s="$set" -v d="$1" '$1 == r && $2 == s && $3 == d { print $5 }' "$results" \
        | sort -u | awk '{ h = $0; n++ } END { print (n == 1 ? substr(h, 1, 16) : "nondeterministic") }'
    }
    b=$(side_hash baseline) h=$(side_hash head)
    if [ "$b" = "$h" ] && [ "$b" != nondeterministic ]; then verdict=yes; else verdict=NO; status=1; fi
    printf '%-10s %-5s %-17s %-17s %-6s %s\n' "$stem" "$set" "$b" "$h" "$verdict" "$runs+$runs"
  done
done
echo "full hashes, sizes and run times: $results"
exit $status
