#!/bin/sh

# Check if the folder path is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_path> <job_script_path>"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Usage: $0 <folder_path> <job_script_path>"
    exit 1
fi

export APEBENCH_ROOT="$WORK/apebench"
export JOB_SCRIPT="$2"

# Recursively find JSON files, excluding those in paths containing STUDY_OUT_*
previous_job_id=""
find "$1" -type f -name "*.json" ! -path "*/STUDY_OUT_*/*" | while read -r json_file; do
    # Get the directory and file name of the current JSON file
    json_dir=$(dirname "$json_file")
    json_name=$(basename "$json_file")

    # Check for STUDY_OUT_* folders in the same directory
    for study_out_dir in "$json_dir"/STUDY_OUT_*; do
        if [ -d "$study_out_dir" ]; then
            # Check for melissa_logger_0.log with "success"
            log_file="$study_out_dir/melissa_server_0.log"
            if [ -f "$log_file" ] && grep -q "Server finalizing with status 0" "$log_file"; then
                # Check if the JSON file name matches any JSON file in STUDY_OUT_*
                if [ -f "$study_out_dir/$json_name" ]; then
                    echo "Skipping $json_file as it matches $study_out_dir/$json_name with success."
                    continue 2 # Skip to the next file in the outer loop
                fi
            fi
        fi
    done

    # Submit sbatch job with a depedency on the previous
    if [ -z "$previous_job_id" ]; then
        # Submit the first job without dependency
        job_id=$(sbatch --parsable $JOB_SCRIPT "$json_file")
    else
        # Submit subsequent jobs with a dependency on the previous job
        job_id=$(sbatch --parsable --dependency=afterok:$previous_job_id $JOB_SCRIPT "$json_file")
    fi

    # Log job submission
    echo "Submitted $json_file as Job ID $job_id"
    previous_job_id=$job_id

done
