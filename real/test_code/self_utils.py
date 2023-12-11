import scipy.io as sio
import os
import numpy as np
import torch
import torch.nn as nn
import logging
from ssim_torch import ssim
from tqdm import tqdm
from torch.autograd import Variable
import scipy
import pywt
from torch.autograd import Function
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import matplotlib.pyplot as plt

from tqdm import tqdm
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous().cpu()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x.cuda()

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2).cpu()

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx.cuda(), None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x.cuda()

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# a=torch.randn(1,31,128,128).cuda()
# b=DWT_2D("haar")(a)
# print(b)


def imshowHSI(img):
    '''

    :param img: B*C*H*W torch B=1
    :return:
    '''

    img=img.detach().cpu().numpy().transpose(1,2,0)
    plt.imshow(img[:,:,[28,14,3]])
    plt.show()

# a=torch.randn(3,256,256).cuda()
#
# imshowHSI(a)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x,is_cuda=True):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
 #########################需要加上cuda########################
    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float()
    if is_cuda==True:
        h=h.cuda()
###############################################
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h, w = 482, 512  # img shape
        self.stride = stride
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_colum = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_Spectral/'
        bgr_data_path = f'{data_root}/Train_RGB/'


        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
        for i in tqdm(range(len(hyper_list))):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
        h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img * self.img_num


class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Valid_Spectral/'
        bgr_data_path = f'{data_root}/Valid_RGB/'
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        for i in tqdm(range(len(hyper_list))):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')


    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)


def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in tqdm(range(len(scene_list))):

        scene_path = path + scene_list[i]
        img_dict = sio.loadmat(scene_path)
        img=img_dict["msi"]
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs
def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 31))
    for i in tqdm(range(len(scene_list))):
        scene_path = path_test + scene_list[i]
        # img = sio.loadmat(scene_path)['msi']
        img=h5py.File(scene_path)["msi"]
        img=np.transpose(img,(1,2,0))
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data
def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)['simulation_test']
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data
def gene_meas(gt_batch):
    spectral_mat = scipy.io.loadmat("cie_1964_w_gain.mat")['filters']
    max=np.max(spectral_mat.ravel())
    min=np.min(spectral_mat.ravel())
    spectral_mat=2*(spectral_mat-min)/(max-min)
    # spectral_mat = scipy.io.loadmat("spectral_mat1.mat")["spectral_mat1"]
    gt_numpy = gt_batch.cpu().detach().numpy()
    gt_numpy1 = np.transpose(gt_numpy, (0, 2, 3, 1))
    gt_numpy2 = np.dot(gt_numpy1, spectral_mat)
    gt_numpy3 = np.transpose(gt_numpy2, (0, 3, 1, 2))
    gt_torch = torch.tensor(gt_numpy3, dtype=torch.float32)
    input_meas = Variable(gt_torch).cuda()
    return input_meas

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_rmse(img,ref):
    # img=(img*256).round()
    # ref=(ref*256).round()
    nC=img.shape[0]
    rmse=0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        rmse+=torch.sqrt(mse)
    return rmse/nC

def torch_mrae(img,ref):
    nC = img.shape[0]
    mrae=0
    for i in range(nC):
        rae=torch.mean(torch.abs(img[i,:,:]-ref[i,:,:])/img[i,:,:])
        mrae+=rae
    return mrae/nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))
def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def shuffle_crop(train_data, batch_size, crop_size=256, argument=True):
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, 31), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))
        # The other half data use splicing.
        processed_data = np.zeros((4, 128, 128, 31), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 31), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch


def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1],256,256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c,h,w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger



def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

#################


import torch
import torch.nn as nn
import logging
import numpy as np
import os

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(model_path, epoch,psnr, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%d_epoch_psrn%.4f.pth' % (epoch,psnr)))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.contiguous().view(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        return torch_psnr(im_true,im_fake)

class Loss_Wavelet(nn.Module):
    def __init__(self):
        super(Loss_Wavelet, self).__init__()
        self.dwt=DWT()
    def forward(self, im_true, im_fake):
        im_true_w=self.dwt(im_true)
        im_fake_w=self.dwt(im_fake)
        # rmse=Loss_MRAE()(im_fake_w,im_true_w)
        error = im_fake_w - im_true_w
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

s=spectral_mat=sio.loadmat("./Ntire2022_response.mat")['filters']
s=torch.from_numpy(s).float().cuda()
class Loss_Spectral(nn.Module):
    def __init__(self):
        super(Loss_Spectral, self).__init__()

    def forward(self, im_true, im_fake):
        im_true_rgb=im_true.permute(0,2,3,1)@s
        im_fake_rgb= im_fake.permute(0,2,3,1)@s

        error = im_fake_rgb - im_true_rgb
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse


# model=Loss_Wavelet()
# a=torch.randn(1,31,128,128)
# b=torch.randn(1,31,128,128)
# temp=Loss_Spectral()(a,b)
# print()
# class Loss_Wavelet(nn.Module):
#     def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
#         super(Loss_Wavelet, self).__init__()
#         self.loss_weight = loss_weight
#         self.alpha = alpha
#         self.patch_factor = patch_factor
#         self.ave_spectrum = ave_spectrum
#         self.log_matrix = log_matrix
#         self.batch_matrix = batch_matrix
#         self.dwt=DWT()
#
#     def tensor2freq(self, x):
#         patch_factor = self.patch_factor
#         _, _, h, w = x.shape
#         assert h % patch_factor == 0 and w % patch_factor == 0, (
#             'Patch factor should be divisible by image height and width')
#         patch_list = []
#         patch_h = h // patch_factor
#         patch_w = w // patch_factor
#         for i in range(patch_factor):
#             for j in range(patch_factor):
#                 patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])
#
#         y = torch.stack(patch_list, 1)
#         y=y.squeeze(1)
#         # y=patch_list
#         # return torch.fft.rfft(y, 2)
#         return self.dwt(y)
#
#     def loss_formulation(self, recon_freq, real_freq, matrix=None):
#         if matrix is not None:
#             weight_matrix = matrix.detach()
#         else:
#             matrix_tmp = (recon_freq - real_freq) ** 2
#             matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha
#             # weight= matrix_tmp / matrix_tmp.max(-1)
#             weight_matrix = matrix_tmp / torch.max(matrix_tmp)
#             weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0)
#         tmp = (recon_freq - real_freq) ** 2
#         freq_distance = tmp[..., 0] + tmp[..., 1]
#
#         loss = weight_matrix * freq_distance
#         return torch.mean(loss)

    # def forward(self, pred, target, matrix=None, **kwargs):
    #
    #     pred_freq = self.tensor2freq(pred)
    #     target_freq = self.tensor2freq(target)
    #
    #     if self.ave_spectrum:
    #         pred_freq = torch.mean(pred_freq, 0, keepdim=True)
    #         target_freq = torch.mean(target_freq, 0, keepdim=True)
    #
    #     return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

# a=torch.randn(5,31,128,128)
# b=torch.randn(5,31,128,128)
#
# model=Loss_Wavelet()
# loss=model(a,b)
# print(loss)





# class Loss_Wavelet(nn.Module):
#     def __init__(self):
#         super(Loss_Wavelet, self).__init__()
#         self.dwt=DWT_2D("haar")
#     def forward(self, im_true, im_fake):
#         im_true_w=self.dwt(self.dwt(im_true))
#         im_fake_w=self.dwt(self.dwt(im_fake))
#         error = im_fake_w - im_true_w
#         sqrt_error = torch.pow(error, 2)
#         rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
#         return rmse
# a=torch.randn(1,31,128,128)
# b=torch.randn(1,31,128,128)
# b=Loss_Wavelet()(a,b)
# print()



        # N = im_true.size()[0]
        # C = im_true.size()[1]
        # H = im_true.size()[2]
        # W = im_true.size()[3]
        # Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        # Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        # mse = nn.MSELoss(reduce=False)
        # err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        # psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        # return torch.mean(psnr)




# a=torch.randn(1,31,256,256)
# b=torch.ones([1,31,256,256])
# c=torch_psnr(a,b)
# d=Loss_PSNR()(a,b)
# print()







class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()

    def forward(self,img,ref):

        ssim_val=0
        for k in range(img.shape[0]):
            ssim_val += torch_ssim(img[k, :, :, :], ref[k, :, :, :])
        return ssim_val/img.shape[0]




def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close