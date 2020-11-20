#! /bin/bash
spinnaker_latency="100"
# 
stages=$1
depth=$2
width=$3

time_scale=$4
input_spike_rate=$5

neurons_per_core=$6
dse_folder="dse_results"

for ms in $spinnaker_latency
do
	sed -i "10s/.*/duration = $ms/" dse_snn_config_LISA_segment

	for ts in $time_scale
	do
		sed -i "28s/.*/time_scale_factor = $ts/" /home/ces10/.spynnaker.cfg

		for npc in $neurons_per_core
		do
			
			sed -i "93s/.*/        self.sim.set_number_of_neurons_per_core(self.sim.SpikeSourcePoisson, $neurons_per_core) # limit number of neurons per core for input spike source DEFAULT 100 IRAKLIS/" /home/ces10/snn_toolbox/snntoolbox/simulation/target_simulators/spinnaker_target_sim.py
			sed -i "108s/.*/        self.sim.set_number_of_neurons_per_core(self.sim.SpikeSourcePoisson, $neurons_per_core) # limit number of neurons per core for input spike source DEFAULT 100 IRAKLIS/" /home/ces10/snn_toolbox/snntoolbox/simulation/target_simulators/spinnaker_target_sim.py
			
			for spike_rate in $input_spike_rate
			do	
				sed -i "231s/.*/            spiketrains = list(rates*$spike_rate) #It was 500 IRAKLIS/" /home/ces10/snn_toolbox/snntoolbox/simulation/target_simulators/spinnaker_target_sim.py
				cd /home/ces10/snn_toolbox
				pip install .
				cd -
				# cd ./LISA_results_motivation/16/2_1_32/
				snntoolbox -t dse_snn_config > ./snn_report_time_scale${ts}_spike_rate${spike_rate}_test.txt
				# snntoolbox -t dse_snn_config > ./snn_reports/snn_report_time_scale${ts}_spike_rate${spike_rate}.txt
				echo $ts
				echo $spike_rate
				cd log/gui/test
				# cp Pearson.png ../../../$dse_folder/$ms/${ts}_${spike_rate}_${neurons_per_core}_pearson.png
				for _dir in *"${pattern}"*; do
				    [ -d "${_dir}" ] && dir="${_dir}" && break
				done
				echo "${dir}"
				cp $dir/5Correlation.png ../../../dse_results_motivation/$ms/${ts}_${spike_rate}_${neurons_per_core}.png
				cd -
				./LISA_cnn_snn_segment_evaluation_args.sh $1 $2 $3 $ts $spike_rate $npc
			done
		done
	done
done
