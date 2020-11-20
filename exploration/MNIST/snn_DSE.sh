#! /bin/bash
spinnaker_latency="100"
time_scale="1"
input_spike_rate="200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500"

neurons_per_core="40"
dse_folder="dse_results_motivation"
dse_reports="dse_reports"

pattern="00Conv2D"

if [ ! -d "$dse_folder" ]; then
	mkdir $dse_folder
fi
if [ ! -d "$dse_reports" ]; then
	mkdir $dse_reports
fi

for ms in $spinnaker_latency
do
	sed -i "10s/.*/duration = $ms/" dse_snn_config
	cd $dse_folder
	if [ ! -d "$ms" ]; then
		mkdir $ms
	fi
	cd ../

	for ts in $time_scale
	do
		sed -i "28s/.*/time_scale_factor = $ts/" /home/ces10/.spynnaker.cfg

		for spike_rate in $input_spike_rate
		do
			sed -i "231s/.*/            nspiketrains = list(rates*$spike_rate) # /" /home/ces10/snn_toolbox/snntoolbox/simulation/target_simulators/spinnaker_target_sim.py
			for npc in $neurons_per_core
			do	
				sed -i "93s/.*/        self.sim.set_number_of_neurons_per_core(self.sim.SpikeSourcePoisson, $neurons_per_core) # limit number of neurons per core for input spike source DEFAULT 100 IRAKLIS/" /home/ces10/snn_toolbox/snntoolbox/simulation/target_simulators/spinnaker_target_sim.py
				sed -i "108s/.*/        self.sim.set_number_of_neurons_per_core(self.sim.SpikeSourcePoisson, $neurons_per_core) # limit number of neurons per core for input spike source DEFAULT 100 IRAKLIS/" /home/ces10/snn_toolbox/snntoolbox/simulation/target_simulators/spinnaker_target_sim.py
				
				cd /home/ces10/snn_toolbox
				pip install .
				cd -
				snntoolbox -t dse_snn_config > ./$dse_reports/snn_report_time_scale${ts}_spike_rate${spike_rate}_neurons_per_core${npc}.txt
				cd log/gui/test
				cp Pearson.png ../../../$dse_folder/$ms/${ts}_${spike_rate}_${neurons_per_core}_pearson.png
				for _dir in *"${pattern}"*; do
				    [ -d "${_dir}" ] && dir="${_dir}" && break
				done
				echo "${dir}"
				cp $dir/5Correlation.png ../../../$dse_folder/$ms/${ts}_${spike_rate}_${neurons_per_core}.png
				cd -
			done
		done
	done
done
