#!/usr/bin/env bash
# =============================================================================
# sync_results.sh
# =============================================================================
# Copy generated artefacts from the WSL working directory to the Windows-side
# project folder, sorted by type.
#
# Code is NOT copied. It moves the other way: Windows folder -> WSL. Copying it
# back risks overwriting an edit with a stale checkout.
#
# Destination layout, under FP2_RRAM_CNN/results/ :
#     csv/        sweep outputs
#     logs/       run logs, including raw NeuroSim output
#     figures/    generated PDF and PNG
#     json/       hardware model reports
#
# USAGE
#     bash sync_results.sh            # copy
#     bash sync_results.sh --dry-run  # list what would be copied
# =============================================================================
set -u

SRC="${SRC:-$HOME/fp2_reram/python}"
DST="${S:-/mnt/c/Users/shaiv/OneDrive - iiit-b/Desktop/IIITB/Projects/FP2_RRAM_CNN}/results"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

if [ ! -d "$SRC" ]; then
    echo "Source not found: $SRC"
    echo "Set SRC=/path/to/working/dir and re-run."
    exit 1
fi

echo "from : $SRC"
echo "to   : $DST"
[ "$DRY" = 1 ] && echo "(dry run)"
echo

copy_group () {          # $1 = subfolder, $2... = find expressions
    local sub="$1"; shift
    local n=0 f
    [ "$DRY" = 0 ] && mkdir -p "$DST/$sub"
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        if [ "$DRY" = 1 ]; then
            echo "  $sub/$(basename "$f")"
        else
            # -u copies only when newer, so re-running is cheap and will not
            # clobber a file already synced.
            cp -u "$f" "$DST/$sub/" 2>/dev/null
        fi
        n=$((n + 1))
    done < <(cd "$SRC" && find . -maxdepth 3 \( -path ./data -o -path ./__pycache__ \) -prune -o \( "$@" \) -print 2>/dev/null | sed "s|^\.|$SRC|")
    printf "  %-10s %d file(s)\n" "$sub" "$n"
}

copy_group csv     -name '*.csv' -type f
copy_group logs    -name '*.log' -type f
copy_group figures -name '*.pdf' -type f -o -name '*.png' -type f
copy_group json    -name '*.json' -type f

echo
if [ "$DRY" = 1 ]; then
    echo "Nothing copied. Re-run without --dry-run."
else
    echo "Done. Totals now in $DST :"
    for d in csv logs figures json; do
        printf "  %-10s %s\n" "$d" "$(ls -1 "$DST/$d" 2>/dev/null | wc -l)"
    done
    echo
    echo "OneDrive syncs on its own; large batches take a few minutes to"
    echo "appear on other machines."
fi
