#!/bin/bash

common_path="experiments/set/"
common_subpath="100BUF_10WM__2TD_8CL/UNet_6_5_relu__decaylr1e-3_1e-4_5000__B256__T75p/"
common_outdir="$HOME/bigfoot_apebench_test/validation_results_decay/"


script_dir="./plot_scripts_all/"
mkdir -p $script_dir

model_ids=("02000" "14000" "21000" "33000")
script_prefix="oar_plot"

oarsub_script="$script_dir/submit_all.sh"
# echo "#!/bin/bash" > $oarsub_script

script_start="#!/bin/bash

#OAR -n melissa-study-plots
#OAR -l /nodes=1/core=16,walltime=00:45:00
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh
singularity_container=\"/bettik/PROJECTS/pr-melissa/COMMON/containers/April23/melissa-active-sampling-with-apebench-cuda.sif\"
"

command_prepend="singularity exec  --bind /bettik:/bettik --env APEBENCH_ROOT=\"$HOME/bigfoot_apebench_test\" \${singularity_container}"

# "python3 plot_model_predictions.py --study-paths $paths --model-id $model_id --all-plots"

# iterate over all the directories in the common_path (one depth level only)
subdirs=$(find ../$common_path -mindepth 1 -maxdepth 1 -type d)
subdirs=$(echo "$subdirs" | grep "diff_")

subdirs=$(echo "$subdirs" | grep -v "diff_ks_cons__3w_x19_easier_1d_x5")

# create all scripts
for dir in $subdirs; do
    dir_basename=$(basename $dir)
    pathhs="${dir}/${common_subpath}*"
    # remove ../ from the path
    pathhs=${pathhs:3}
    # add \$APEBENCH_ROOT/ to the path
    pathhs="$HOME/bigfoot_apebench_test/$pathhs"
    for model_id in "${model_ids[@]}"; do
        script_name="${script_prefix}_${model_id}_${dir_basename}.sh"
        script_path="${script_dir}${script_name}"
        # take the last 4 characters of the model_id
        model_id_=${model_id: -4}
        # echo "Creating script: $script_path"
        echo "$script_start" > $script_path
        echo "$command_prepend python3 $HOME/bigfoot_apebench_test/common/plot_model_predictions.py --study-paths $pathhs --model-id $model_id_ --all-plots --output-dir $common_outdir " >> $script_path
        chmod +x $script_path
        # echo "oasrsub -S $script_path" >> $oarsub_script
    done
done
echo "#!/bin/bash" > $oarsub_script
# list all the scripts as in sorted order
for script in $(ls -1 $script_dir | sort -V | grep $script_prefix); do
    echo "oarsub -S ./$script" >> $oarsub_script
done

chmod +x $oarsub_script
echo "All scripts created in $script_dir"
echo "All oarsub scripts created in $oarsub_script"
