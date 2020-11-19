#! /bin/bash
image_size="16"
stages="2"
depth="1"
width="32"
experiment_folder="FMNIST_results"
project_dir=$1
cd $project_dir

if [ ! -d "$experiment_folder" ]; then
	mkdir $experiment_folder
fi

cd $experiment_folder

for size in $image_size
do
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
				dir=$s
				dir+=_
				dir+=$d
				dir+=_
				dir+=$w
				if [ ! -d "$dir" ]; then 
					mkdir "$dir"
					cd "$dir"
					mkdir dataset
					cd ../../../
				else
					cd ../../	
				fi
				if [ ! -d "$experiment_folder/$size/$dir/dataset" ]; then 
					cd "$experiment_folder/$size/$dir"
					mkdir  dataset
					cd ../../../
				fi
				python CNN_FMNIST.py "$size" "$s" "$d" "$w" > cnn_report_FMNIST.txt
				mv cnn_report_FMNIST.txt FMNIST_cnn.h5 $experiment_folder/$size/$dir/
				mv x_norm.npz x_test.npz y_test.npz $experiment_folder/$size/$dir/dataset
				
				cp dse_snn_config_FMNIST $experiment_folder/$size/$dir/dse_snn_config

				cp snn_DSE.sh $experiment_folder/$size/$dir/
				cd $experiment_folder/$size/$dir/
				
				snntoolbox -t dse_snn_config > snn_report_FMNIST.txt
				# ./snn_DSE.sh
				cd ../
			done
		done
	done
	cd ../
done

