#! /bin/bash
calc(){ awk "BEGIN { print "$*" }"; }
# DANGE: IMAGESIZE 
image_size="16"
stages=$1
depth=$2
width=$3
ts=$4
spike_rate=$5

project_dir=$6
experiment_folder=$project_dir/"FMNIST_results_Giannis"

if [ ! -d "$experiment_folder" ]; then
	mkdir $experiment_folder
fi

# dataset_segment_dir="dataset_segment"
snn_reports_dir="snn_reports"


for size in $image_size
do
	cd $experiment_folder
	if [ ! -d "$size" ]; then # Control will enter here if $DIRECTORY doesn't exist.
		mkdir $size
	fi
	cd "$size"

	for s in $stages
	do	
		for d in $depth
		do
			for w in $width
			do
				cd $experiment_folder/$size

				dir=$s
				dir+=_
				dir+=$d
					dir+=_
				dir+=$w
				if [ ! -d "$dir" ]; then 
					mkdir "$dir"
					cd "$dir"
					echo "Created: "$dir
					mkdir dataset
					# cd ../../../
					cd $experiment_folder
				else
					cd $experiment_folder	
				fi
				experiment_dir=$experiment_folder/$size/$dir/
				cd $experiment_dir
				if [ ! -d "$experiment_folder/$size/$dir/dataset" ]; then 
					cd "$experiment_folder/$size/$dir"
					mkdir  dataset
					cd ../../../
				fi

				if [ ! -d "$snn_reports_dir" ]; then 
					mkdir $snn_reports_dir
				fi
				cd $project_dir	

				cd $experiment_folder/$size/$dir/
				snntoolbox -t dse_snn_config > ./$snn_reports_dir/snn_report_time_scale${ts}_spike_rate${spike_rate}.txt
				rm -rf ./reports/
			done
		done
	done
	# cd ../
done
