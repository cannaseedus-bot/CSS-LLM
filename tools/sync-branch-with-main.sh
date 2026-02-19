#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-origin}"
BASE_BRANCH="${2:-main}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository" >&2
  exit 1
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [[ -z "$CURRENT_BRANCH" ]]; then
  echo "Detached HEAD; checkout your feature branch first" >&2
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Remote '$REMOTE' is not configured." >&2
  echo "Add it first: git remote add $REMOTE <repo-url>" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is dirty. Commit or stash changes before syncing." >&2
  exit 3
fi

echo "Fetching $REMOTE..."
git fetch "$REMOTE"

echo "Rebasing $CURRENT_BRANCH onto $REMOTE/$BASE_BRANCH..."
git rebase "$REMOTE/$BASE_BRANCH"

echo "Done. If this branch was already pushed, run:"
echo "  git push --force-with-lease $REMOTE $CURRENT_BRANCH"
