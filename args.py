import argparse

def str2bool(t):
    if t.lower() in ['true', 't', '1']:
        return True
    else:
        return False

def get_args():

    parser = argparse.ArgumentParser(description='CS 395T: Grounded Natural Language Processing')

    # General
    parser.add_argument('--mode', default='extract_captions', type=str, help='mode of the python script')

    # DataLoader
    parser.add_argument('--data_dir', default='./data', type=str, help='root directory of the dataset')
    parser.add_argument('--corpus', default='msvd_vgg', type=str, help='video captioning corpus to use')
    parser.add_argument('--nworkers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--bsize', default=32, type=int, help='mini-batch size')
    parser.add_argument('--shuffle', default='True', type=str2bool, help='shuffle the data?')
    parser.add_argument('--glove_emb_file', default='glove.6B.300d.txt', type=str, help='pretrained glove embedding file')

    # Image Model Parameters
    parser.add_argument('--img_size', default=224, type=int, help='image size for the CNN model')
    parser.add_argument('--vision_arch', default='resnet34', type=str, help='base ImageNet model for video features')
    parser.add_argument('--num_frames', default=30, type=int, help='number of frames to extract from the video')
    parser.add_argument('--vid_feat_size', default=4096, type=int, help='size of the video features')

    # Model Parameters
    parser.add_argument('--arch', default='s2vt', type=str, help='video captioning model architecture')
    parser.add_argument('--max_len', default=20, type=int, help='max length of the sentence')
    parser.add_argument('--dropout_p', default=0.2, type=float, help='dropout probability')
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden layer size')
    parser.add_argument('--schedule_sample', default=False, type=str2bool, help='perform schedule sampling while training')
    parser.add_argument('--tau', default=1.0, type=float, help='Temperature parameter for Gumbel-Softmax')
    parser.add_argument('--pretrained_base', default=None, type=str, help='Pretrained video captioning model')

    # Optimization Parameters
    parser.add_argument('--optim', default='adam', type=str, help='optimizer type')
    parser.add_argument('--lr', default=2e-3, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--max_norm', default=1, type=float, help='max grad norm')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--lambda_brev', default=1.0, type=float, help='Scaling factor for brevity loss')
    parser.add_argument('--lambda_cont', default=1.0, type=float, help='Scaling factor for continuity loss')

    # Other
    parser.add_argument('--save_path', default='./trained_models', type=str, help='directory where models are saved')
    parser.add_argument('--log_dir', default='./logs', type=str, help='directory where tensorboardX logs are saved')
    parser.add_argument('--log_iter', default=5, type=int, help='print frequency')
    parser.add_argument('--n_sample_sent', default=5, type=int, help='number of sample sentences to show during validation')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume if previous checkpoint exists')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training')

    args = parser.parse_args()

    print('Running on {} corpus'.format(args.corpus.upper()))
    if args.corpus not in ['msvd', 'msrvtt', 'msvd_vgg']:
        raise NotImplementedError('Unknown corpus')

    return args
