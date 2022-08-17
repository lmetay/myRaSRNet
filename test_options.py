import argparse
parser = argparse.ArgumentParser(description='Test Change Detection Models')

####------------------------------------   ttsting parameters   --------------------------------------####

parser.add_argument('--test_batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')


####----------------------------------   path for loading data   -----------------------------------------####
parser.add_argument('--test1_dir', default='./LEVIR256/A', type=str, help='t1 image path for testing')
parser.add_argument('--test2_dir', default='./LEVIR256/B', type=str, help='t2 image path for testing')
parser.add_argument('--label_test', default='./LEVIR256/label', type=str, help='label path for testing')


####----------------------------   network loading and result saving   ------------------------------------####
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
parser.add_argument('--name', type=str, default='LEVIR', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--load_net', type=str, default='RaSRNet_test_LEVIR.pth',
                        help='name of the experiment. It decides where to store samples and models') 
parser.add_argument('--results_dir', type=str, default='./result/', help='saves results here.')

####-------------------------------------   Model settings   -----------------------------------------####
parser.add_argument('--in_c', default=3, type=int, help='input channel')
parser.add_argument('--out_cls', default=2, type=int, help='output category')
parser.add_argument('--filter_num', type=list, default=[256,128,64,32], help='the output dimension')
