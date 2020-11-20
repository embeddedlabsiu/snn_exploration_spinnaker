#! /bin/bash
image_size="16"
stages="2"
depth="1"
width="6"
experiment_folder="MNIST_results_Giannis"

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

				python CNN_MNIST.py "$size" "$s" "$d" "$w" > cnn_report_MNIST.txt
				mv cnn_report_MNIST.txt MNIST_cnn.h5 $experiment_folder/$size/$dir/
				mv x_norm.npz x_test.npz y_test.npz $experiment_folder/$size/$dir/dataset
				cp dse_snn_config_MNIST $experiment_folder/$size/$dir/dse_snn_config
				cp snn_DSE.sh $experiment_folder/$size/$dir/
				cd $experiment_folder/$size/$dir/
				./snn_DSE.sh
				cd ../
			done
		done
	done
	cd ../
done
