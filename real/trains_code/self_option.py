import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='Reconstruction lab',
                    help='You can set various templates in option.py')

# Hardware specifications

parser.add_argument("--gpu_id", type=str, default='2')

# Data specifications
parser.add_argument('--data_root', type=str, default='/data1/lxx_data', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='/', help='saving_path')

# Model specifications

parser.add_argument('--method', type=str, default='ChannlBasedRecurmer', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications


parser.add_argument("--crop_size",type=int,default=128,help="crop size")
parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=1000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--stride", type=int, default=8, help="stride")
# parser.add_argument("--outf", type=str, default='./mst_plus_plus/', help='path log files')
# PATH6='net_model_path/state_dict_SSRNet.pth'
opt = parser.parse_args()
template.set_template(opt)

# dataset
opt.data_path = f"{opt.data_root}/Train Spectral/"
opt.test_path = f"{opt.data_root}/Valid Spectral/"


for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False