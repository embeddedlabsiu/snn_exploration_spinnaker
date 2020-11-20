#! /bin/bash
spinnaker_latency="100"
# 
config_file="dse_snn_config_FMNIST"
images_to_test=20
simulator="spinnaker"
sed -i "9s/.*/simulator = $simulator/" $config_file
sed -i "11s/.*/num_to_test = $images_to_test/" $config_file

# cat $config_file

stages=$1
depth=$2
width=$3

time_scale=$4
input_spike_rate=$5

neurons_per_core="100"
dse_folder="dse_results"

# figure_path="./log/gui/test/00Conv2D_4x16x16" 
for ms in $spinnaker_latency
do
	sed -i "10s/.*/duration = $ms/" dse_snn_config_FMNIST

	for ts in $time_scale
	do
		sed -i "28s/.*/time_scale_factor = $ts/" /home/ces10/.spynnaker.cfg

		for spike_rate in $input_spike_rate
		do
			sed -i "231s/.*/            spiketrains = list(rates*$spike_rate) #It was 500 IRAKLIS/" /home/ces10/snn_toolbox/snntoolbox/simulation/target_simulators/spinnaker_target_sim.py
			cd /home/ces10/snn_toolbox
			pip install .
			cd -
			# snntoolbox -t dse_snn_config > ./snn_reports/snn_report_time_scale${ts}_spike_rate${spike_rate}.txt
			echo $ts
			echo $spike_rate
			./FMNIST_cnn_snn_evaluation_args.sh $1 $2 $3 $ts $spike_rate
		done
	done
done
