#!/bin/bash

# we create another script with OAR settings to fit the right time
# argument 1 is a folder which we need to recursively get to config files
# we save absolute paths to the config files in a variable
# then we get the number of the config files N, len of the array
# and calculate total time needed to run all the config files, which will be  N * 60 minutes
# then we loop over the config files
# and run the melissa-launcher command for each config file

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
# Get the absolute path of the given folder
abs_path=$(realpath "$given_folder")

script_name="${abs_path}/bigf_set.sh"
# Check if the script already exists
if [ -f "$script_name" ]; then
	echo "The script $script_name already exists. Please remove it or choose a different name."
	exit 1
fi

# Find all config files in the given folder and its subdirectories without offline in the name
config_files=($(find "$abs_path" -type f -name "config*mpi.json" ! -name "*offline*"))
# Check if any config files were found
if [ ${#config_files[@]} -eq 0 ]; then
	echo "No config files found in the given folder."
	exit 1
fi
echo "The folder $abs_path contains ${#config_files[@]} config files."
# Create the script file
echo "#!/bin/bash" > "$script_name"

# Get the number of config files
num_config_files=${#config_files[@]}
# Calculate the total time needed to run all config files
total_time=$((num_config_files * 1))
# Print the total time needed
echo "Total time needed to run all config files: $total_time hours"

first_echo="
#OAR -n melissa-study-bench
#OAR -l /nodes=1/core=10/gpu=1,walltime=$total_time:00:00
#OAR -p gpumodel='V100'
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh
singularity_container=\"/bettik/PROJECTS/pr-melissa/COMMON/containers/April23/melissa-active-sampling-with-apebench-cuda.sif\"

"

echo "$first_echo" >> "$script_name"

# Loop over the config files and write to script the melissa-launcher command for each
for config_file in "${config_files[@]}"; do
	config_path=$(realpath "$config_file")
	config_file_name=$(basename "$config_file")
	echo "Processing config file: $config_file_name"
	echo "singularity exec --nv --bind /bettik:/bettik --env APEBENCH_ROOT=\"\$HOME/apebench_test\" \${singularity_container} melissa-launcher --config_name $config_path" >> "$script_name"
done

# Make the script executable
chmod +x "$script_name"
echo "Script $script_name created successfully."
echo "You can run it with: oarsub -S ./$script_name"
echo "Note: Make sure to check the script for any errors before running it."