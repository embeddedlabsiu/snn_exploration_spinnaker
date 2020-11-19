#! /bin/bash
calc(){ awk "BEGIN { print "$*" }"; }
# DANGE: IMAGESIZE 
image_size="16"
stages=$1
depth=$2
width=$3

# absolute paths of the experiment -> avoid path errors 
project_dir="/home/project"
experiment_folder=$project_dir/"LISA_results"

if [ ! -d "$experiment_folder" ]; then
	mkdir $experiment_folder
fi

dataset_segment_dir="dataset_segment"
snn_reports_dir="snn_reports_neurons_per_core"


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
				cp dse_snn_config_LISA_segment $experiment_folder/$size/$dir/dse_snn_config
				sum=0

				total_segments=11

				# change directory to the experiment directory once (use absolute paths to copy or move files)
				cd $experiment_dir
				for ((segment=1; segment<=total_segments;segment++))
				do
					# copy .npz files from the parent directory to the local dataset_segment
					cp $project_dir/LISA_segments/"$size"/x_test_${segment}.npz $experiment_dir/$dataset_segment_dir
					cp $project_dir/LISA_segments/"$size"/y_test_${segment}.npz $experiment_dir/$dataset_segment_dir
					cp $project_dir/LISA_segments/"$size"/x_norm.npz $experiment_dir/$dataset_segment_dir
					

					# change name to default so that snntoolbox receives the correct x_test.npz, y_test.npz,  
					mv ./$dataset_segment_dir/x_test_${segment}.npz ./$dataset_segment_dir/x_test.npz 
					mv ./$dataset_segment_dir/y_test_${segment}.npz ./$dataset_segment_dir/y_test.npz
					
					# change the filename: $4 = time scale, $5 = spike rate
					snntoolbox -t dse_snn_config > ./$snn_reports_dir/snn_report_time_scale_${4}_spike_rate_${5}_neurons_per_core${6}_segment_${segment}.txt
	  
					# remove files before starting new iteration for different segment
					rm ./$dataset_segment_dir/x_test.npz 
					rm ./$dataset_segment_dir/y_test.npz 
					rm ./$dataset_segment_dir/x_norm.npz
					
					parse and convert the percentage of the number of images found per segment					
					# tail -n <5> might need to change because the snntoolbox changes the output file
					string_to_parse=$(tail -n 14 ./snn_reports/snn_report_time_scale_${1}_spike_rate_${2}_segment_${segment}.txt | grep accuracy)
					num=$(echo -e $string_to_parse | tr '\n' ' ' | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g' | tr -s ' ' | sed 's/ /\n/g')
					item=(${num[0]})
					images=$(($item / 10))
				    sum=$(( $sum + $images ))  
					cd -
				done
				cd $experiment_dir
				total_images=$(($total_segments * 10))
				acc=$(calc $sum/$total_images)

				# save accuracy into local file
				echo "Images found: "$sum, "Total images: "$total_images ,"Time Scale:" $1, "Spike Rate: "$2, "Accuracy: "$acc >> accuracy.txt
				
			done
		done
	done
done
