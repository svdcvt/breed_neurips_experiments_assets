#!/bin/bash

# we create another script with OAR settings to fit the right time
# argument 1 is a folder which we need to recursively get to config files
# we save absolute paths to the config files in a variable
# then we get the number of the config files N, len of the array
# and calculate total time needed to run all the config files, which will be  N * 60 minutes
# then we loop over the config files
# and run the melissa-launcher command for each config file
# example of running the script > separate script in 5 scripts, each config is 30 minutes run
# ./make_job_set.sh path/to/folder/ 5 30


given_folder="$1"
if [ -z "$given_folder" ]; then
	echo "Please provide a folder containing the config files."
	exit 1
fi
if [ ! -d "$given_folder" ]; then
	echo "The provided path is not a directory."
	exit 1
fi

number=$2
if [ -z "$number" ]; then
	echo "Please provide a number for the script name."
	exit 1
fi
if ! [[ "$number" =~ ^[0-9]+$ ]]; then
	echo "The provided number is not valid. Please provide a positive integer."
	exit 1
fi

minutes=$3
if [ -z "$minutes" ]; then
	minutes=70
elif ! [[ "$minutes" =~ ^[0-9]+$ ]]; then
	echo "The provided minutes is not valid. Please provide a positive integer."
	exit 1
fi

# Get the absolute path of the given folder
abs_path=$(realpath "$given_folder")

script_name="${abs_path}/bigf_set"
for i in $(seq 1 $number); do
	ii=$((i - 1))
	script_name_="${script_name}_${ii}.sh"
	# Check if the script already exists
	if [ -f "$script_name_" ]; then
		echo "The script $script_name_ already exists. Please remove it or choose a different name."
		exit 1
	fi
done

# Find all config files in the given folder and its subdirectories without offline in the name
config_files=($(find "$abs_path" -type f -name "config*mpi.json" ! -name "*offline*"))
# Check if any config files were found
if [ ${#config_files[@]} -eq 0 ]; then
	echo "No config files found in the given folder."
	exit 1
fi
echo "The folder $abs_path contains ${#config_files[@]} config files."

# Create the script file
for i in $(seq 1 $number); do
	ii=$((i - 1))
	script_name_="${script_name}_${ii}.sh"
	echo "#!/bin/bash" > "$script_name_"
done

# Get the number of config files
num_config_files=${#config_files[@]}
# Calculate the total time needed to run all config files
total_time=$((num_config_files * minutes / number))
# Convert total time to hours and minutes
total_time_hours=$((total_time / 60))
total_time_minutes=$((total_time % 60))
# Convert total to HH:MM format
if [ $total_time_hours -lt 10 ]; then
	total_time_hours="0$total_time_hours"
fi
if [ $total_time_minutes -lt 10 ]; then
	total_time_minutes="0$total_time_minutes"
fi
total_time="$total_time_hours:$total_time_minutes"
# Print the total time needed
echo "Total time needed to run all config files: $total_time hours"

# first part of the cluster job submission script TODO:ANONYMYSE
first_echo="
#OAR -n melissa-study-bench
#OAR -l /nodes=1/core=10/gpu=1,walltime=$total_time:00
#OAR -p gpumodel='V100'
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh
singularity_container=\"/bettik/PROJECTS/pr-melissa/COMMON/containers/April23/melissa-active-sampling-with-apebench-cuda.sif\"

"

for i in $(seq 1 $number); do
	ii=$((i - 1))
	script_name_="${script_name}_${ii}.sh"
	echo "$first_echo" >> "$script_name_"
done

# Loop over the config files and write to script the melissa-launcher command for each
for i in $(seq 1 $num_config_files); do
	script_i=$((i % number))
	script_name_="${script_name}_${script_i}.sh"
	# Get the config file for this script
	config_file=${config_files[$((i - 1))]}
	config_path=$(realpath "$config_file")
	config_file_name=$(basename "$config_file")
	echo "Processing config file: $config_file_name"
	echo "singularity exec --nv --bind /bettik:/bettik --env APEBENCH_ROOT="$HOME/apebench_test" \${singularity_container} melissa-launcher --config_name $config_path" >> "$script_name_"
	echo "sleep 10" >> "$script_name_"
done

# Make the script executable
for i in $(seq 1 $number); do
	ii=$((i - 1))
	script_name_="${script_name}_${ii}.sh"
	# Make the script executable
	chmod +x "$script_name_"
done
echo "Script $script_name created successfully."
echo "You can run it with: oarsub -S ./$script_name"
echo "Note: Make sure to check the script for any errors before running it."