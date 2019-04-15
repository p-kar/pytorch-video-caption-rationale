source /scratch/cluster/pkar/pytorch-gpu-py3/bin/activate
code_root=__CODE_ROOT__

python -u $code_root/driver.py \
	--mode __MODE__ \
	--data_dir __DATA_DIR__ \
	--corpus __CORPUS__ \
	--nworkers __NWORKERS__ \
	--bsize __BSIZE__ \
	--shuffle __SHUFFLE__ \
	--glove_emb_file __GLOVE_EMB_FILE__ \
	--img_size __IMG_SIZE__ \
	--vision_arch __VISION_ARCH__ \
	--num_frames __NUM_FRAMES__ \
	--vid_feat_size __VID_FEAT_SIZE__ \
	--arch __ARCH__ \
	--max_len __MAX_LEN__ \
	--dropout_p __DROPOUT_P__ \
	--hidden_size __HIDDEN_SIZE__ \
	--schedule_sample __SCHEDULE_SAMPLE__ \
	--optim __OPTIM__ \
	--lr __LR__ \
	--wd __WD__ \
	--momentum __MOMENTUM__ \
	--epochs __EPOCHS__ \
	--max_norm __MAX_NORM__ \
	--start_epoch __START_EPOCH__ \
	--save_path __SAVE_PATH__ \
	--log_dir __LOG_DIR__ \
	--log_iter __LOG_ITER__ \
	--n_sample_sent __N_SAMPLE_SENT__ \
	--resume __RESUME__ \
	--seed __SEED__