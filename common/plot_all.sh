#!/bin/bash

# will time overall how long it took to run
start_time=$(date +%s)
common_path="../experiments/set/"
common_subpath="100BUF_10WM__2TD_8CL/UNet_6_5_relu__constlr1e-03_B256__T75p/"
# iterate over all the directories in the common_path (one depth level only)
subdirs=$(find $common_path -mindepth 1 -maxdepth 1 -type d)
# exclude that contains "ks__3w_x07_harder_max1_1d"
subdirs=$(echo "$subdirs" | grep -v "ks__3w_x07_harder_max1_1d")
# exclude "kdv__"
subdirs=$(echo "$subdirs" | grep -v "kdv__")
echo "Subdirectories found:"
echo "$subdirs"
# iterate over all the directories in the subdirs
for dir in $subdirs; do
    pathhs="${dir}/${common_subpath}*"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Plotting for $dir"
    # echo "<<<<<< Model 2000"
    # # echo time current
    # echo "Current time: $(date +%T)"
    # python3 plot_model_predictions.py --study-paths $pathhs --model-id 2000
    # echo "<<<<<< Model 4000"
    # echo "Current time: $(date +%T)"
    # python3 plot_model_predictions.py --study-paths $pathhs --model-id 4000
    echo "<<<<<< Model BEST"
    echo "Current time: $(date +%T)"
    python3 plot_model_predictions.py --study-paths $pathhs
done
# will time overall how long it took to run
end_time=$(date +%s)
# calculate the time taken
time_taken=$((end_time - start_time))
# print the time taken in hours, minutes and seconds
hours=$((time_taken / 3600))
minutes=$(( (time_taken % 3600) / 60 ))
seconds=$((time_taken % 60))
echo "Time taken to run the script: $hours hours, $minutes minutes and $seconds seconds"
