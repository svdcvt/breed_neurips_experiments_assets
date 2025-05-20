#!/bin/bash

common_path="experiments/set/"
common_subpath="100BUF_10WM__2TD_8CL/UNet_6_5_relu__decaylr1e-3_1e-4_5000__B256__T75p/"
common_outdir="$REPO_ROOT/validation_results/validation_results_decay/"


script_dir="$REPO_ROOT/plot_scripts_all/"
mkdir -p $script_dir

model_ids=("02000" "14000" "21000" "33000") # first digit is to give it an order 
script_prefix="job_plot"

job_script="$script_dir/submit_all.sh"

script_start="#!/bin/bash

YOUR_CLUSTER_SCHEDULER_SETTINGS (16 CPU cores, 0 GPU, 45 minutes)

singularity_container=\"$REPO_ROOT/melissa-active-sampling-with-apebench-cuda.sif\"

"

command_prepend="singularity exec  --env REPO_ROOT=\"$REPO_ROOT\" \${singularity_container}"

# "python3 plot_model_predictions.py --study-paths $paths --model-id $model_id --all-plots"

# iterate over all the directories in the common_path (one depth level only)
subdirs=$(find ../$common_path -mindepth 1 -maxdepth 1 -type d)
subdirs=$(echo "$subdirs" | grep "diff_")

# create all scripts
for dir in $subdirs; do
    dir_basename=$(basename $dir)
    pathhs="${dir}/${common_subpath}*"
    # remove ../ from the path
    pathhs=${pathhs:3}
    pathhs="$REPO_ROOT/$pathhs"
    for model_id in "${model_ids[@]}"; do
        script_name="${script_prefix}_${model_id}_${dir_basename}.sh"
        script_path="${script_dir}${script_name}"
        # take the last 4 characters of the model_id
        model_id_=${model_id: -4}
        echo "$script_start" > $script_path
        echo "$command_prepend python3 $REPO_ROOT/common/plot_model_predictions.py --study-paths $pathhs --model-id $model_id_ --all-plots --output-dir $common_outdir " >> $script_path
        chmod +x $script_path
    done
done

echo "#!/bin/bash" > $job_script
# list all the scripts as in sorted order
for script in $(ls -1 $script_dir | sort -V | grep $script_prefix); do
    echo "oarsub -S ./$script" >> $job_script
done

chmod +x $job_script
echo "All scripts created in $script_dir"
echo "All oarsub scripts created in $job_script"
