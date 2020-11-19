#! /bin/bash
calc(){ awk "BEGIN { print "$*" }"; }
# DANGE: IMAGESIZE 
image_size="16"
stages=$1
depth=$2
width=$3

# absolute paths of the experiment -> avoid path errors 
project_dir=$6
experiment_folder=$project_dir/"FMNIST_results"

if [ ! -d "$experiment_folder" ]; then
	mkdir $experiment_folder
fi

dataset_segment_dir="dataset_segment"
snn_reports_dir="snn_reports"


for size in $image_size
do
	cd $experiment_folder
	if [ ! -d "$size" ]; then 
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
					cd $experiment_folder
				else
					cd $experiment_folder	
				fi
				experiment_dir=$experiment_folder/$size/$dir/
				cd $experiment_dir
				if [ ! -d "$dataset_segment_dir" ]; then 
					mkdir $dataset_segment_dir
				fi
				if [ ! -d "$snn_reports_dir" ]; then 
					mkdir $snn_reports_dir
				fi
				cd $project_dir	
				python CNN_FMNIST.py "$size" "$s" "$d" "$w" > cnn_report_FMNIST.txt
				mv cnn_report_FMNIST.txt FMNIST_cnn.h5 $experiment_dir
				cp dse_snn_config_FMNIST_segment $experiment_folder/$size/$dir/dse_snn_config
				sum=0

				total_segments=5

				# change directory to the experiment directory once (use absolute paths to copy or move files)
				cd $experiment_dir
				for ((segment=1; segment<=total_segments;segment++))
				do
					# copy .npz files from the parent directory to the local dataset_segment
					cp $project_dir/FMNIST_segments/"$size"/x_test_${segment}.npz $experiment_dir/$dataset_segment_dir
					cp $project_dir/FMNIST_segments/"$size"/y_test_${segment}.npz $experiment_dir/$dataset_segment_dir
					cp $project_dir/FMNIST_segments/"$size"/x_norm.npz $experiment_dir/$dataset_segment_dir
					

					# change name to default so that snntoolbox receives the correct x_test.npz, y_test.npz,  
					mv ./$dataset_segment_dir/x_test_${segment}.npz ./$dataset_segment_dir/x_test.npz 
					mv ./$dataset_segment_dir/y_test_${segment}.npz ./$dataset_segment_dir/y_test.npz
					
					# change the filename: $4 = time scale, $5 = spike rate
					snntoolbox -t dse_snn_config > ./snn_reports/snn_report_time_scale_${4}_spike_rate_${5}_segment_${segment}.txt
	  
					# remove files before starting new iteration for different segment
					rm ./$dataset_segment_dir/x_test.npz 
					rm ./$dataset_segment_dir/y_test.npz 
					rm ./$dataset_segment_dir/x_norm.npz
				done
			done
		done
	done
done
