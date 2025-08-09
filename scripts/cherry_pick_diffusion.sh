#!/usr/bin/env bash
set -euo pipefail

# Cherry-pick (checkout) diffusion-related files from feature/diffusion into target branches.
#
# Usage:
#   scripts/cherry_pick_diffusion.sh [-n] [-p] branch1 [branch2 ...]
#
# Options:
#   -n  Dry-run (show actions without changing branches or files)
#   -p  Push after committing to each branch (uses current upstream)
#
# Defaults:
#   If no branches are provided, the script exits with usage help.

DRY_RUN=0
DO_PUSH=0
while getopts ":np" opt; do
  case $opt in
    n) DRY_RUN=1 ;;
    p) DO_PUSH=1 ;;
    *) echo "Usage: $0 [-n] [-p] <branch> [branch...]" >&2; exit 2 ;;
  esac
done
shift $((OPTIND-1))

if [[ $# -lt 1 ]];
then
  echo "Usage: $0 [-n] [-p] <branch> [branch...]" >&2
  exit 2
fi

ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$ROOT_DIR"

SOURCE_BRANCH="feature/diffusion"

FILES=(
  "models/diffusion/ddim.py"
  "models/diffusion/hrm_denoiser.py"
  "models/diffusion/__init__.py"
  "scripts/diffuse_bytes.py"
  "tests/test_diffusion_determinism.py"
  "models/layers.py"  # includes SDPA fallback for attention
  "cibvo.md"
)

echo "[info] Repo: $ROOT_DIR"
echo "[info] Source branch: $SOURCE_BRANCH"
echo "[info] Targets: $*"
echo "[info] Dry-run: $DRY_RUN  Push: $DO_PUSH"

# Basic validation
if ! git rev-parse --verify "$SOURCE_BRANCH" >/dev/null 2>&1; then
  echo "[error] Source branch not found: $SOURCE_BRANCH" >&2
  exit 1
fi

# Ensure files exist on source branch
for f in "${FILES[@]}"; do
  if ! git cat-file -e "$SOURCE_BRANCH:$f" 2>/dev/null; then
    echo "[warn] Missing on $SOURCE_BRANCH: $f"
  fi
done

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

for target in "$@"; do
  echo "\n[task] Updating: $target"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  would: git switch $target"
    for f in "${FILES[@]}"; do
      echo "  would: git checkout $SOURCE_BRANCH -- $f"
    done
    echo "  would: git add <files> && git commit -m 'chore(diffusion): import scheduler+denoiser from $SOURCE_BRANCH'"
    [[ $DO_PUSH -eq 1 ]] && echo "  would: git push"
    continue
  fi

  # Switch
  git fetch --all --quiet
  git switch "$target"

  # Checkout files from source branch
  UPDATED=0
  for f in "${FILES[@]}"; do
    if git cat-file -e "$SOURCE_BRANCH:$f" 2>/dev/null; then
      git checkout "$SOURCE_BRANCH" -- "$f"
      UPDATED=1
    else
      echo "  [skip] not on $SOURCE_BRANCH: $f"
    fi
  done

  if [[ $UPDATED -eq 1 ]]; then
    git add -A
    if ! git diff --cached --quiet; then
      git commit -m "chore(diffusion): import scheduler+denoiser from $SOURCE_BRANCH"
      [[ $DO_PUSH -eq 1 ]] && git push
      echo "  [ok] committed to $target"
    else
      echo "  [noop] no changes to commit on $target"
    fi
  else
    echo "  [noop] no files updated on $target"
  fi
done

# Return to original branch if not in dry-run
if [[ $DRY_RUN -eq 0 ]]; then
  git switch "$CURRENT_BRANCH" >/dev/null 2>&1 || true
fi

echo "\n[done] Review commits on target branches."
