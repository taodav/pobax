: '
Uses onager to prelaunch all experiments in a given directory.
'
PRELAUNCH_DIR='hyperparams/gvd_frozen_mem/annas_maze'
RAN_JOBS=''

for item in $PRELAUNCH_DIR/*;
do
  if [[ -f $item ]]; then
    echo
    echo "---------Prelaunching $item---------"
    python onager_write_jobs.py $item
    JOB_NAME=$(basename $item .py)
    RAN_JOBS="$RAN_JOBS $JOB_NAME"
  else
    echo
    echo "Skipped $item"
  fi
done

echo "List of job names ran:"
echo $RAN_JOBS