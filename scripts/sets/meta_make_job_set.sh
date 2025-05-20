#!/bin/bash

# # This script is used to create a set of job scripts for a given folder containing config files.
# # It will find all the directories that match a given filter and create a job script for each of them.
# # It will also create a meta job script that will run all the job scripts in parallel.
# # The script will delete the job scripts if the third argument is set to "delete".

# # example of running the script:
# # .meta_make_job_set.sh /path/to/folder/of/folders/ "filter_string"
# # example of deleteing the job scripts:
# # .meta_make_job_set.sh /path/to/folder/of/folders/ "filter_string" delete
# # example filter matching x OR y
# # .meta_make_job_set.sh /path/to/folder/of/folders/ "x|y"
# # example filter matching x AND y
# # .meta_make_job_set.sh /path/to/folder/of/folders/ "x.*y"


given_folder="$1"
if [ -z "$given_folder" ]; then
	echo "Please provide a folder containing the config files."
	exit 1
fi
if [ ! -d "$given_folder" ]; then
	echo "The provided path is not a directory."
	exit 1
fi

filter=$2
if [ -z "$filter" ]; then
	echo "Please provide a filter string."
	exit 1
fi

flag_delete=$3

abs_path=$(realpath "$given_folder")

# find recursively such directories that end with the filter string
filtered_directories=($(find "$abs_path" -type d -name "*$filter"))
# how many directories
num_filtered_directories=${#filtered_directories[@]}
echo "The folder $abs_path contains ${num_filtered_directories} directories with the filter $filter."

# iterate over the directories and run make_job_set with that directory and number 1
if [ -z "$flag_delete" ]; then
    for folder in "${filtered_directories[@]}"; do
        echo "Running make_job_set for $folder"
        # make_job_set.sh <directory> <number> <minutes>
        minutes=70
        substringnot70="diff_ks__[23]w_default"
        # check if this string in the name and if yes then minutes is 130
        if echo "$folder" | grep -q "$substringnot70"; then
            minutes=130
        fi

        scripts_path=$(dirname "$(realpath "$0")")
        "$scripts_path/make_job_set.sh" "$folder" 1 $minutes
    done
fi
# than gather all the scripts names in a file that will do oarsub -S script_name
# scripts made by make_job_set.sh are named job_set_0.sh inside the folder
# and the script that will do oarsub -S is named job_set_meta_{filter}.sh
    script_name="${abs_path}/job_set_meta_${filter}.sh"
if [ -z "$flag_delete" ]; then
    echo "#!/bin/bash" > "$script_name"
else
    echo "Deleting the script $script_name"
    rm "$script_name"
fi

for folder in "${filtered_directories[@]}"; do
    # get the script name
    script_name_="${folder}/job_set_0.sh"
    # check if the script exists
    if [ -f "$script_name_" ]; then
        if [ -z "$flag_delete" ]; then
            echo "oarsub -S $script_name_" >> "$script_name"
        else
            echo "Deleting the script $script_name_"
            rm "$script_name_"
        fi
    else
        echo "The script $script_name_ does not exist."
    fi
done
# make the script executable
if [ -z "$flag_delete" ]; then
    chmod +x "$script_name"
    echo "The script $script_name has been created and is executable."
fi
