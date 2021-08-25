for num_fc_layers in {1,2,3,4}; do echo "
	#PBS -l walltime=24:00:00
	#PBS -l select=1:ncpus=16:mem=96gb:ngpus=4:gpu_type=RTX6000

	module load anaconda3/personal
	source activate py38
	cd /rds/general/user/yh2520/ephemeral/Simultaneous-Sound-Localisation-Transformer/src

	python3 model_train.py \"./Spectral_1600K_2Sound_std_1208/\" \"./model/2108_2Sound_tf_src_$num_fc_layers/\" 16 "allRegression" "transformer" 2 \"src\" --numEnc 3 --numFC $num_fc_layers --batchSize 256 --numEpoch 30 --valDropout 0.1 --lrRate 1e-4 --Ncues 4 --isHPC \"T\" --valDropout 0.1 --coordinates \"spherical\"
	" > submit_train_2108_$num_fc_layers.pbs; qsub submit_train_2108_$num_fc_layers.pbs; done;
