#! /bin/bash
spinnaker_latency="100"
# 
stages=$1
depth=$2
width=$3

time_scale=$4
input_spike_rate=$5

neurons_per_core="100"
dse_folder="dse_results"

for ms in $spinnaker_latency
do
	sed -i "10s/.*/duration = $ms/" dse_snn_config_MNIST_segment

	for ts in $time_scale
	do
		sed -i "28s/.*/time_scale_factor = $ts/" /home/ces10/.spynnaker.cfg

		for spike_rate in $input_spike_rate
		do
			sed -i "231s/.*/            spiketrains = list(rates*$spike_rate) " /home/ces10/snn_toolbox/snntoolbox/simulation/target_simulators/spinnaker_target_sim.py
			cd /home/ces10/snn_toolbox
			pip install .
			cd -

			./MNIST_cnn_snn_segment_evaluation_args.sh $1 $2 $3 $ts $spike_rate
		done
	done
done
