import os
import pdb
import subprocess as sp

OUTPUT_ROOT='/scratch/cluster/pkar/pytorch-video-caption-rationale/runs/train_s2vt_att_msvd_vgg'
SCRIPT_ROOT='/scratch/cluster/pkar/pytorch-video-caption-rationale/scripts'

mapping_dict = {
    # Condor Scheduling Parameters
    '__EMAILID__': 'pkar@cs.utexas.edu',
    '__PROJECT__': 'INSTRUCTIONAL',
    # Script parameters
    '__JOBNAME__': ['lr_1e-4', 'lr_1e-3'],
    # Algorithm hyperparameters
    '__CODE_ROOT__': '/scratch/cluster/pkar/pytorch-video-caption-rationale',
    '__MODE__': 'train',
    '__DATA_DIR__': '/scratch/cluster/pkar/pytorch-video-caption-rationale/data',
    '__CORPUS__': 'msvd_vgg',
    '__NWORKERS__': '4',
    '__BSIZE__': '32',
    '__SHUFFLE__': 'true',
    '__GLOVE_EMB_FILE__': 'glove.6B.300d.txt',
    '__IMG_SIZE__': '224',
    '__VISION_ARCH__': 'resnet34',
    '__NUM_FRAMES__': '30',
    '__VID_FEAT_SIZE__': '4096',
    '__ARCH__': 's2vt-att',
    '__MAX_LEN__': '20',
    '__DROPOUT_P__': '0.4',
    '__HIDDEN_SIZE__': '512',
    '__SCHEDULE_SAMPLE__': 'false',
    '__OPTIM__': 'adam',
    '__LR__': ['1e-4', '1e-3'],
    '__WD__': '5e-4',
    '__MOMENTUM__': '0.9',
    '__EPOCHS__': '1000',
    '__MAX_NORM__': '1',
    '__START_EPOCH__': '0',
    '__LOG_ITER__': '5',
    '__N_SAMPLE_SENT__': '5',
    '__RESUME__': 'true',
    '__SEED__': '123',
}

# Figure out number of jobs to run
num_jobs = 1
for key, value in mapping_dict.items():
    if type(value) == type([]):
        if num_jobs == 1:
            num_jobs = len(value)
        else:
            assert(num_jobs == len(value))

for idx in range(num_jobs):
    job_name = mapping_dict['__JOBNAME__'][idx]
    mapping_dict['__LOGNAME__'] = os.path.join(OUTPUT_ROOT, job_name)
    if os.path.isdir(mapping_dict['__LOGNAME__']):
        print ('Skipping job ', mapping_dict['__LOGNAME__'], ' directory exists')
        continue

    mapping_dict['__LOG_DIR__'] = mapping_dict['__LOGNAME__']
    mapping_dict['__SAVE_PATH__'] = mapping_dict['__LOGNAME__']
    sp.call('mkdir %s'%(mapping_dict['__LOGNAME__']), shell=True)
    condor_script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'condor_script.sh')
    script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'run_script.sh')
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'condor_script_proto.sh'), condor_script_path), shell=True)
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'run_proto.sh'), script_path), shell=True)
    for key, value in mapping_dict.items():
        if type(value) == type([]):
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], condor_script_path), shell=True)
        else:
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, condor_script_path), shell=True)

    sp.call('condor_submit %s'%(condor_script_path), shell=True)
