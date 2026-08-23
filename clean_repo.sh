#!/usr/bin/env bash
# =============================================================================
# clean_repo.sh
# =============================================================================
# Remove redistributable-restricted and regenerable files from the repository.
#
# 41 publisher PDFs are committed and pushed. They were obtained under an
# institutional subscription, and IEEE, Elsevier and Springer licences do not
# permit republication, which is what a public repository is. They account for
# most of the 77 MB history.
#
# Stage 1 removes them from the index and from future commits. Stage 2 rewrites
# history so earlier commits no longer carry them, and needs a force push.
#
# Run from WSL or Git Bash, not PowerShell.
#
# USAGE
#     bash clean_repo.sh --stage1     # index only, reversible
#     bash clean_repo.sh --stage2     # history rewrite, force push required
# =============================================================================
set -eu
cd "$(dirname "$0")"

# A stale index.lock makes every write operation fail. Left unchecked with
# `|| true` around each git call, stage 1 reports success and changes nothing.
LOCK=".git/index.lock"
if [ -e "$LOCK" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
    echo "$LOCK exists, ${age}s old."
    echo
    echo "If no git command and no editor is running, it is stale. On a folder"
    echo "synced by OneDrive it is often the sync client holding the file."
    echo "Pause syncing, then:"
    echo "    rm -f '$PWD/$LOCK'"
    exit 1
fi

untrack () {                 # $1 = path, only if git is tracking it
    if git ls-files --error-unmatch "$1" >/dev/null 2>&1 \
       || [ -n "$(git ls-files "$1")" ]; then
        git rm -r --cached --quiet "$1"
        echo "  untracked $1"
    else
        echo "  $1 already untracked"
    fi
}

case "${1:-}" in
--stage1)
    echo "Stage 1: index only."
    untrack references
    untrack python
    untrack fp2-reram-crossbar.zip
    # INDEX.md records what each paper supplies. Original work, so it stays.
    [ -f references/INDEX.md ] && git add -f references/INDEX.md
    git add .gitignore
    [ -d results ] && git add results
    for f in clean_repo.sh sync_results.sh rerun_neurosim.sh; do
        [ -f "$f" ] && git add "$f"
    done
    git add -u -- '*.py' '*.tex' '*.md' 2>/dev/null || true
    echo
    staged=$(git diff --cached --name-only | wc -l)
    echo "$staged path(s) staged. Review:  git status"
    if [ "$staged" -eq 0 ]; then
        echo "Nothing staged. Something blocked the writes -- do not commit."
        exit 1
    fi
    echo "Commit:  git commit -m 'Track results; untrack papers and working tree'"
    echo
    echo "Files stay on disk; only tracking changes. Earlier commits still"
    echo "carry the PDFs until stage 2 runs."
    ;;

--stage2)
    # git-filter-repo may be a git subcommand, a bare executable, or importable
    # only as a module. pip on Windows often installs the .exe outside PATH.
    if git filter-repo --help >/dev/null 2>&1; then
        FR() { git filter-repo "$@"; }
    elif command -v git-filter-repo >/dev/null 2>&1; then
        FR() { git-filter-repo "$@"; }
    elif python3 -c "import git_filter_repo" >/dev/null 2>&1; then
        FR() { python3 -m git_filter_repo "$@"; }
    elif python -c "import git_filter_repo" >/dev/null 2>&1; then
        FR() { python -m git_filter_repo "$@"; }
    else
        echo "git-filter-repo not found on PATH or as a module."
        echo "    pip install --user git-filter-repo"
        echo "On Windows pip installs the .exe outside PATH; the module form"
        echo "    python -m git_filter_repo"
        echo "works regardless and is what this script prefers."
        exit 1
    fi

    echo "Stage 2 rewrites every commit. Hashes change; the remote needs a"
    echo "force push. Pause OneDrive first: it syncs .git file by file and can"
    echo "corrupt a rewrite in progress."
    read -r -p "Proceed? [y/N] " a
    [ "$a" = "y" ] || { echo "Aborted."; exit 0; }

    FR --force --invert-paths \
       --path references --path python --path fp2-reram-crossbar.zip

    echo
    echo "Done. filter-repo drops the remote by design. Restore and push:"
    echo "    git remote add origin https://github.com/ShaiviNandi/FP2_RRAM_CNN.git"
    echo "    git push --force origin main"
    echo
    echo "Existing clones keep the old objects, so treat the PDFs as having"
    echo "been public."
    ;;

*)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
