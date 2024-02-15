#! /bin/bash

# This script assumes that we have initialized the appropriate python
# environment ahead of time and that we just need to execute the various Python
# scripts in order.

# The raters we are running over:
VENTRAL_RATERS=(bogengsong BrendaQiu JiyeongHa lindazelinzhao nourahboujaber jennifertepan)
DORSAL_RATERS=(Annie-lsc BrendaQiu mominbashir oadesiyan qiutan6li)
MEANRATER=mean
# The input and output directories:
SAVE_PATH=/data/crcns2021/results/data_branch/save
PROC_PATH=/data/crcns2021/results/proc

# Die function for errors:
function die {
    echo "$*"
    exit 1
}

# Make sure we're in the repository directory above the scripts directory.
SCRIPT_DIR=`cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd`
cd "$SCRIPT_DIR"/..
[ -d ./hcpannot ] && [ -d ./scripts ] \
    || die "script must be in the hcp-annot-vc repo when run"

# Figure out which region we are processing.
region="$1"
if [ "$1" = "ventral" ]
then RATERS=(${VENTRAL_RATERS[@]})
elif [ "$1" = "dorsal" ]
then RATERS=(${DORSAL_RATERS[@]})
else die "Syntax: proc_all.sh [ventral|dorsal] [options]"
fi
shift

# (1) Process all the individual raters.
python scripts/proc_raters.py \
    "$region" \
    "$SAVE_PATH" "$PROC_PATH" \
    "$@" --raters ${RATERS[@]}
# (2) Process the means.
python scripts/proc_means.py \
    "$region" \
    "$PROC_PATH"/traces "$PROC_PATH" \
    "$@" --raters ${RATERS[@]}

# That's it!
exit 0
