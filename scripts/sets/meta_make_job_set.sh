#!/bin/bash

#first to make sure we need to see if we are on bigfoot frontend
if [ "$(hostname)" != "bigfoot" ]; then
	echo "This script should be run on the bigfoot frontend."
	exit 1
fi


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
        # minutes=70
        minutes=35
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
# scripts made by make_job_set.sh are named bigf_set_0.sh inside the folder
# and the script that will do oarsub -S is named bigf_set_meta_{filter}.sh
    script_name="${abs_path}/bigf_set_meta_${filter}.sh"
if [ -z "$flag_delete" ]; then
    echo "#!/bin/bash" > "$script_name"
else
    echo "Deleting the script $script_name"
    rm "$script_name"
fi

for folder in "${filtered_directories[@]}"; do
    # get the script name
    script_name_="${folder}/bigf_set_0.sh"
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
#UNet_6_5_relu__decaylr1e-3_1e-4_5000__B256__T75p

# example of running the script:
# ./scripts/sets/meta_make_job_set.sh /path/to/folder/of/folders/ "filter_string"
# example filter matching x OR y
# ./scripts/sets/meta_make_job_set.sh /path/to/folder/of/folders/ "x|y"
# example filter matching x AND y
# ./scripts/sets/meta_make_job_set.sh /path/to/folder/of/folders/ "x.*y"


