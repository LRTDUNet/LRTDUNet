import numpy as np
import  h5py
from Process.LineShow import *
from Process.RGBShow import *
from tqdm import tqdm
import os
import pandas as pd
import matplotlib
import scipy.io as scio
# matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
# data=scio.loadmat("./orig.mat")["truth"]
# plt.imshow(data[:,:,26],cmap="gray")
# plt.show()
# print()
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


markers = ['^', 'D', '*', '|', '.', '1', '2', '3', '4', 'v','o','<']
# bands = [453.5,457.5,462.0,466.0,471.5,476.5,481.5,487.0,492.5,498.0,504.0,510.0,
#             516.0,522.5,529.5,536.5,544.0,551.5,558.5,567.5,575.5,584.5,594.5,604.0,
#             614.5,625.0,636.5,648.0]
bands = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610
    , 620, 630, 640, 650, 660, 670, 680, 690, 700]


def patch(img, min_x, min_y, width, height):
    return img[min_y:min_y + height, min_x:min_x + width, :]
    # return img[min_y:min_y+height,min_x:min_x+width]


def drawInImg(img, roi, min_x, min_y, width, height):
    img_ = img.copy()
    # mask = cv2.resize(roi, (100, 100), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    # img_[0:100, 0:100] = mask
    # cv2.rectangle(img_, (0, 0), (100, 100), (255, 255, 255), 2)
    # //在图像上绘制矩形
    cv2.rectangle(img_, (min_x, min_y), (min_x + width, min_y + height), (255, 255, 255), 2)
    return img_

def drawRGB(img, roi, min_x, min_y, width, height):
    img_ = img.copy()
    # mask = cv2.resize(roi, (100, 100), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    # img_[0:100, 0:100] = mask
    # cv2.rectangle(img_, (0, 0), (100, 100), (255, 255, 255), 2)
    cv2.rectangle(img_, (min_x, min_y), (min_x + width, min_y + height), (255, 255, 255), 2)
    return img_

def drawRectangle(img, min_x, min_y, width, height):
    # mask = cv2.resize(roi, (100, 100), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    # img_[0:100,0:100]=mask
    # cv2.rectangle(img_,(0,0),(100,100),(255,255,255),2)
    img2 = img.copy()
    cv2.rectangle(img2, (min_x, min_y), (min_x + width, min_y + height), (255, 255, 255), 2)
    # cv2.rectangle(img,(min_x,min_y),(min_x+width,min_y+height),(255,255,255),2)
    return img2

def local(method_list, file_path, num):
    data = []

    ref = sio.loadmat("./orig.mat")['truth']#ARAD_1k_0913
    # ref = h5py.File("./data4/orig.mat")['cube']
    # ref = np.array(ref)
    # ref = np.transpose(ref, (2, 1, 0))
    # ref=ref[:256,:256,:]
    # # ref=ref[]
    ######################################
    # with h5py.File("./data1/orig.mat", 'r') as mat:
    #     ref = np.float32(np.array(mat['cube']))
    #     ref=np.transpose(ref,(2,1,0))
    # plt.imshow(ref[:,:,12],cmap="gray")
    # plt.show()
    # ref=h5py.File("./data1/orig.mat")
    # ref = sio.loadmat("./data1/orig.mat")['truth']
    # ref=np.transpose(ref,(1,2,0))
    # ref = sio.loadmat("./data/screne04.mat")['msi']


    ################

    # ref = h5py.File("./")["msi"]
    # ref = np.array(ref).transpose(2,1,0)

    data.append(ref)
    for path in tqdm(file_path):
        #仿真数据 标签是msi  在 “./data/” 文件夹下的
        res = sio.loadmat(path)['msi']
        # res=res[:256,:256,:]
        # res = sio.loadmat(path)['truth']
        # res=np.transpose(res,(1,0,2))
        data.append(res)
    # plt.imshow(data[2][:,:,2])
    # plt.show()

    for i in range(num):
        # ref_img = ref[i][:,:,(10,14,20)]
        ref_img = ref[:, :, [11, 18, 24]]
        min_x, min_y, width, height = roi_draw(ref_img,50,50)

        for idx in range(len(method_list)):
            psnr = torch_psnr(patch(data[idx], min_x, min_y, width, height), patch(data[0], min_x, min_y, width, height))
            # ssim=torch_ssim(patch(data[idx], min_x, min_y, width, height), patch(data[0], min_x, min_y, width, height))
            print("Method:{} ,psnr{}".format(method_list[idx], psnr))
            print("Method:{} ,ssim{}".format(method_list[idx], ssim))
        #     ##############################
            for j in [0,8,16,24,30]:
            # for j in range(0,31):
                img = data[idx]
                for times in range(3):
                    ref_ = frame2rgb(img[:, :, j], bands[j])
                    ref_roi_ = frame2rgb(patch(img, min_x, min_y, width, height)[:, :, j], bands[j])
                method_name = method_list[idx]
                save_path = './Test12KKzoom/{}/sence_{}/'.format(method_name, i)
                if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
                    os.makedirs(save_path)
                temp = drawInImg(ref_, ref_roi_, min_x, min_y, width, height)
                temp_rgb=drawRGB(ref_img,ref_roi_, min_x,min_y, width, height)
                # temp2=frame2rgb(ref_roi_,ba)
                temp2=ref_roi_

                cv2.imwrite('./Test12KKzoom/{}/sence_{}/roi_{}.png'.format(method_name, i, bands[j]), temp2 * 255)
                cv2.imwrite('./Test12KKzoom/{}/sence_{}/rgb_.png'.format(method_name, i), temp_rgb * 255)
                cv2.imwrite('./Test12KKzoom/{}/sence_{}/{}.png'.format(method_name, i, bands[j]), temp * 255)
                cv2.imwrite('./Test12KKzoom/{}/sence_{}/rgb.png'.format(method_name, i), temp_rgb * 255)
                print("finish")


# def localreal(method_list, file_path, num):
#     data = []
#     ref = sio.loadmat("./orig.mat")['truth']
#     # data.append(ref)
#     for path in tqdm(file_path):
#         res = sio.loadmat(path)['msi']
#         data.append(res)
#
#
#     for i in range(num):
#         ref_img = data[0][i][:, :, (10, 14, 20)]
#         min_x, min_y, width, height = roi_draw(ref_img, 100, 100)
#         for idx in range(len(method_list)):
#             for j in [11, 18, 25]:
#                 img = data[idx][i]
#                 ref_ = frame2rgb(img[:, :, j], bands[j])
#                 ref_roi_ = frame2rgb(patch(img, min_x, min_y, width, height)[:, :, j], bands[j])
#                 method_name = method_list[idx]
#                 save_path = './zoom_Real/{}/sence_{}/'.format(method_name, i)
#                 if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
#                     os.makedirs(save_path)
#                 temp = drawInImg(ref_, ref_roi_, min_x, min_y, width, height)
#                 cv2.imwrite('./zoom_Real/{}/sence_{}/{}.png'.format(method_name, i, bands[j]), temp * 255)

def cruve(method_list, file_path, num):
    ##绘制曲线的时候ref 不加如data里面
    data = []
    method = method_list[0:]
    num_method = len(method)
    # ref = sio.loadmat(file_path[0])['truth']
    # 下面是机器人数据
    # ref = sio.loadmat("./orig.mat")["truth"]
    # ref = sio.loadmat("./data1/orig.mat")[""]
    # //仿真数据
    ref = sio.loadmat("./data/screne04.mat")["msi"]
    # ref = h5py.File("./data4/orig.mat")['cube']
    # ref = np.array(ref)

    # ref=
    # ref = np.transpose(ref, (2, 1, 0))
    # ref = ref[:256, :256, :]
    # #########################################
    # data.append(ref)
    for path in tqdm(file_path):
        # res = sio.loadmat(path)['pred']
        res = sio.loadmat(path)['msi']
        data.append(res)
###################################
    # ref = h5py.File("./scenes/scene03.mat")["msi"]
    # ref = np.array(ref).transpose(2,1,0)
    #
    # # data.append(ref)
    # for path in tqdm(file_path):
    #     #仿真数据 标签是msi  在 “./data/” 文件夹下的
    #     res = sio.loadmat(path)['msi']
    #     # res = sio.loadmat(path)['truth']
    #     res=np.transpose(res,(1,0,2))
    #     data.append(res)


    temp_list=[]
    tempa=0
    for i in range(num):
        for times in range(3):
            # ref_img=ref[i]
            # ref_img = ref[i][:,:,(10,14,20)]
            ref_img = ref[:, :, (10, 14, 20)]
            rgb = ref_img
            # for times in range(3)://20
            min_x, min_y, width, height = roi_draw(rgb, 20, 20)
            rgb = ref_img
            temp = drawRectangle(rgb, min_x, min_y, width, height)
            temp_list.append(temp)

            # plt.imshow(temp,cmap="gray")
            # plt.show()
                # cv2.rectangle(rgb,(min_x,min_y),(min_x+width,min_y+height),(255,255,255),2)
            line_ref = norm(mean_cal(patch(ref, min_x, min_y, width, height), 31))
            plt.plot(bands, line_ref, marker=markers[0], markersize=6, color='red')
            # plt.show()

            relations = [
                line(patch(img, min_x, min_y, width, height), patch(ref, min_x, min_y, width, height), 31) for img
                in data
            ]
            for rc in range(len(relations)):
                if rc == (len(relations) - 1):
                    plt.plot(bands, relations[rc][1], marker=markers[rc + 1], markersize=6, color='black')
                else:
                    plt.plot(bands, relations[rc][1],marker=markers[rc+1], markersize=6)
                    # plt.plot(bands,relations[rc][1],marker=markers[rc+1],markersize=1.5)

            plt.legend(['ref'] + [
                method[idx] + ",corr:%5f" % relations[idx][0] for idx in range(num_method)])
            # plt.legend(fontsize=5)
            plt.ylabel("Density", fontsize=14)
            plt.xlabel("Wavelength (nm)", fontsize=10)
            # save_path = './Test62922curves/sence_{}/ROI_{}/'.format(i, times)
            save_path = './Test12KKzoom/sence_{}/ROI_{}/'.format(i, times)
            if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
                os.makedirs(save_path)
            plt.savefig(save_path + 'ROI_{}.svg'.format(times), format='svg')
            cv2.imwrite(save_path + 'ROI_{}.png'.format(times), temp * 255)
            plt.clf()
        #############################################
        for times in range(len(temp_list)):
            tempa+=temp_list[times]
        tempa/=len(temp_list)
                # temp+=temp_list[times]
        cv2.imwrite(save_path + 'ROI_all{}.png'.format(times), tempa * 255)


def showAllBands( ):
    # ref=sio.loadmat(path_root+"/ours.mat")["msi"]
    ref = sio.loadmat(path_root)["msi"]
    for i in range(31):

        # plt.imshow(ref[:,:,i],cmap="gray")
        temp = frame2rgb(ref[:, :, i], bands[i])
        # save_path = './showbands/bands_{}/'.format(i)
        save_path="./showbands20/"
        if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
            os.makedirs(save_path)
        # plt.savefig(save_path + '.png')
        cv2.imwrite(save_path + '{}.png'.format(i), temp * 255)
    print()


import torch

def torch_psnr(img, ref):  # input [28,256,256]
    # img=torch.from_numpy(img)
    img = (img*256).round()
    # ref=torch.from_numpy(ref)
    ref = (ref*256).round()
    nC = img.shape[2]
    psnr = 0
    for i in range(nC):
        mse = np.mean((img[:, :, i] - ref[:, :, i]) ** 2)
        psnr += 10 * np.log10((255*255)/mse)
    return psnr / nC


def torch_ssim(img, ref):  # input [28,256,256]
    img=torch.from_numpy(img).float()
    ref=torch.from_numpy(ref).float()
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def showPSNR(method_list,file_path):
    data = []

    ref = h5py.File(path_root+"/orig.mat")['cube']
    ref=np.array(ref)
    ref=np.transpose(ref,(2,1,0))
    data.append(ref)
    for path in tqdm(file_path):
        #仿真数据 标签是msi  在 “./data/” 文件夹下的
        res = sio.loadmat(path)['msi']
        # res = sio.loadmat(path)['truth']
        data.append(res)
    for idx in range(len(method_list)):
        psnr=torch_psnr(ref,data[idx])
        # mrae=torch_mrae(ref,data[idx])
        # rmse=torch_rmse(ref,data[idx])
        print("Method:{} ,psnr{}".format(method_list[idx],psnr))
        # print("Method:{} ,mrae{}".format(method_list[idx], mrae))
        # print("Method:{} ,rmse{}".format(method_list[idx], rmse))


def torch_rmse(img,ref):
    # img=(img*256).round()
    # ref=(ref*256).round()
    nC=img.shape[2]
    rmse=0
    for i in range(nC):
        mse = np.mean((img[:, :, i] - ref[:, :, i]) ** 2)
        rmse+=np.sqrt(mse)
    return rmse/nC

def torch_mrae(img,ref):
    nC = img.shape[2]
    mrae=0
    for i in range(nC):
        rae=np.mean(np.abs(img[:,:,i]-ref[:,:,i])/img[:,:,i])
        mrae+=rae
    return mrae/nC


def gene_meas(gt_batch):
    #input b ,nc ,h,w
    ###########cie 1964's result
    spectral_mat = scio.loadmat("../cie_1964_w_gain.mat")['filters']
    max=np.max(spectral_mat.ravel())
    min=np.min(spectral_mat.ravel())
    spectral_mat=2*(spectral_mat-min)/(max-min)
    ####NTIRE2022 的结果##############################
    # spectral_mat=scio.loadmat("Ntire2022_reponse.mat")['filters']
    ###############################################################

    # spectral_mat = scipy.io.loadmat("spectral_mat1.mat")["spectral_mat1"]
    # b,nC,h,w=gt_batch.shape
    gt_numpy = gt_batch.cpu().detach().numpy()
    gt_numpy1 = np.transpose(gt_numpy, (0, 2, 3, 1))
    gt_numpy2 = np.dot(gt_numpy1, spectral_mat)

    gt_numpy3 = np.transpose(gt_numpy2, (0, 3, 1, 2))
    gt_torch = torch.tensor(gt_numpy3, dtype=torch.float32)
    input_meas = Variable(gt_torch)

    return input_meas

def GeneRGB(METHOD,file_path):
    data=[]
    # ref = h5py.File(path_root+"/orig.mat")['cube']
    # ref = np.array(ref)
    # ref = np.transpose(ref, (2, 1, 0))
    # data.append(ref)
    # ref=h5py.File(path_root+"/orig.mat")['cube']
    ref = h5py.File(path_root + "/orig.mat")['msi']
    ref=np.transpose(ref,(0,2,1))
    ref = np.array(ref)
    rgb = gene_meas(torch.from_numpy(ref).unsqueeze(0)).float()
    rgb = rgb.squeeze(0)
    rgb = rgb.detach().cpu().numpy()
    rgb = np.transpose(rgb, (2, 1, 0))
    # rgb=np
    data.append(rgb)

    for path in tqdm(file_path):
        #仿真数据 标签是msi  在 “./data/” 文件夹下的
        res = sio.loadmat(path)['msi']
        res=np.transpose(res,(2,0,1))
        ####（31,3）
        # res=torch.transpose(torch.from_numpy(res),(2,0,1))
        rgb = gene_meas(torch.from_numpy(res).unsqueeze(0)).float()
        rgb=rgb.squeeze(0)
        rgb=rgb.detach().cpu().numpy()
        rgb=np.transpose(rgb,(1,2,0))
        data.append(rgb)
    # img = np.transpose(img, (0, 2, 1))
    # rgb = gene_meas(torch.from_numpy(img).unsqueeze(0)).float()
    # x = rgb.cpu()
    save_path = "./scene06_rgb/"
    if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
        os.makedirs(save_path)
    for i in range(len(METHOD)):
        temp=data[i]
        max = np.max(temp.ravel())
        min = np.min(temp.ravel())
        temp = 0.5 * (temp - min) / (max - min)
        # temp=np.transpose(temp,(1,0,2))
        # plt.imshow(temp[:,:,[0,1,2]],cmap="gray")
        # plt.show()
        cv2.imwrite(save_path + '{}.png'.format(METHOD[i]), temp * 255)
        # 计算RGB 的psnr

def cal_PSRN_RGB(file_path,method_list):
    ref=plt.imread(path_root+"/orig.png")
    # data=[]
    for i in range(len(method_list)) :
        img=plt.imread(file_path[i])
        # data.append()
        psnr = torch_psnr(ref, img)
        ssim = torch_ssim(ref,img)
        print("Method:{} ,psnr{}".format(method_list[i], psnr))
        print("Method:{} ,ssim{}".format(method_list[i], ssim))

def cruze_RGB(file_path,method_list):
    ##绘制曲线的时候ref 不加如data里面
    data = []
    method = method_list[0:]
    num_method = len(method)
    ref = plt.imread(path_root + "/orig.png")
    for i in range(len(method_list)):
        img = plt.imread(file_path[i])
        data.append(img)
    temp_list = []
    # tempa = 0
    # for i in range(num):
    for times in range(3):
            # ref_img=ref[i]
            # ref_img = ref[i][:,:,(10,14,20)]
            ref_img = ref[:, :,:]
            rgb = ref_img
            # for times in range(3)://20
            min_x, min_y, width, height = roi_draw(rgb, 50, 50)
            rgb = ref_img
            temp = drawRectangle(rgb, min_x, min_y, width, height)
            temp_list.append(temp)
            # plt.imshow(temp,cmap="gray")
            # plt.show()
            # cv2.rectangle(rgb,(min_x,min_y),(min_x+width,min_y+height),(255,255,255),2)
            line_ref = norm(mean_cal(patch(ref, min_x, min_y, width, height), 3))
            plt.plot([400,500,600], line_ref, marker=markers[0], markersize=3, color='red')
            # plt.show()

            relations = [
                line(patch(img, min_x, min_y, width, height), patch(ref, min_x, min_y, width, height), 3) for img
                in data
            ]
            for rc in range(len(relations)):
                if rc == (len(relations) - 1):
                    plt.plot([400,500,600], relations[rc][1], marker=markers[rc + 1], markersize=5, color='black')
                else:
                    plt.plot([400,500,600], relations[rc][1],marker=markers[rc+1], markersize=1.5)
            # for rc in range(len(relations)):
            #     if rc == (len(relations) - 1):
            #         plt.plot(bands, relations[rc][1], marker=markers[rc + 1], markersize=6, color='black')
            #     else:
            #         plt.plot(bands, relations[rc][1], marker=markers[rc + 1], markersize=6)


            plt.legend(['ref'] + [
                method[idx] + ",corr:%5f" % relations[idx][0] for idx in range(num_method)])
            # plt.legend(fontsize=5)
            plt.ylabel("Density", fontsize=14)
            plt.xlabel("Wavelength (nm)", fontsize=10)
            save_path = './Testrgb11curves/sence_{}/ROI_{}/'.format(i, times)
            if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
                os.makedirs(save_path)
            plt.savefig(save_path + 'ROI_{}.svg'.format(times), format='svg')
            cv2.imwrite(save_path + 'ROI_{}.png'.format(times), temp * 255)
            plt.clf()


        #############################################
    # for times in range(len(temp_list)):
    #     tempa += temp_list[times]
    #     tempa /= len(temp_list)
    #     # temp+=temp_list[times]
    #     cv2.imwrite(save_path + 'ROI_all{}.png'.format(times), tempa * 255)

def rgb_local(method_list, file_path, num):
    data = []
    method = method_list[0:]
    num_method = len(method)
    ref = plt.imread(path_root + "/orig.png")
    data.append(ref)
    for i in range(len(method_list)-1):
        img = plt.imread(file_path[i])
        data.append(img)
    temp_list = []
    for i in range(num):
        # ref_img = ref[i][:,:,(10,14,20)]
        ref_img = ref[:, :,:]
        min_x, min_y, width, height = roi_draw(ref_img,50,50)

        for idx in range(len(method_list)):
            psnr = torch_psnr(patch(data[idx], min_x, min_y, width, height), patch(data[0], min_x, min_y, width, height))
            ssim=torch_ssim(patch(data[idx], min_x, min_y, width, height), patch(data[0], min_x, min_y, width, height))
            print("Method:{} ,psnr{}".format(method_list[idx], psnr))
            print("Method:{} ,ssim{}".format(method_list[idx], ssim))
        #     ##############################
        #     for j in [0,8,16,24,30]:
            # for j in range(0,31):
            img = data[idx]
            for times in range(3):
                # ref_roi_=img[min_x:min_x+width,min_y: min_y+height,:]
                ref_roi_=patch(img,min_x,min_y,width,height)
                    # ref_ = frame2rgb(img[:, :, j], bands[j])
                    # ref_roi_ = frame2rgb(patch(img, min_x, min_y, width, height)[:, :, j], bands[j])
                method_name = method_list[idx]
                save_path = './SimulathionRGBzoom33/{}/sence_{}/'.format(method_name, i)
                if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
                    os.makedirs(save_path)
                temp = drawInImg(img, ref_roi_, min_x, min_y, width, height)
                # temp_rgb=drawRGB(ref_img,ref_roi_, min_x,min_y, width, height)
                # temp2=frame2rgb(ref_roi_,ba)
                # temp2=ref_roi_
                # cv2.imwrite('./SimulathionRGBzoom3/{}/sence_{}/roi_.png'.format(method_name, i), ref_roi_ * 255)
                # cv2.imwrite('./SimulathionRGBzoom3/{}/sence_{}/rgb_.png'.format(method_name, i),  temp* 255)
                # # cv2.imwrite('./Test13zoom/{}/sence_{}/rgb_.png'.format(method_name, i), temp_rgb * 255)
                # # cv2.imwrite('./Test13zoom/{}/sence_{}/{}.png'.format(method_name, i, bands[j]), temp * 255)
                # # cv2.imwrite('./Test13zoom/{}/sence_{}/rgb.png'.format(method_name, i), temp_rgb * 255)
                # print("finish")


def RGBTOHSIShow():
    ref=plt.imread("./grayscale_image.png")
    for i in range(31):
        temp = frame2rgb(ref, bands[i])
        save_path = "./WaveRGBshow/"
        if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
            os.makedirs(save_path)

        cv2.imwrite(save_path + '{}.png'.format(i), temp*255)





if __name__ == "__main__":
    # RGBTOHSIShow()
    # showAllBands()
    # showAllBands()
    # path_root="./scene01_rgb"
    # path_root="./scene06_rgb"
    # path_root="./genergb"
    # path_root="../data_20/ours.mat"
    # path_root="./data4" ###机器人数据
    path_root = "./modelOut" ###仿真数据
    # path_root = "./data28_rgb
    # path_root="./data4_rgb"
    # path_root="./scene03_rgb/"
    # GRB_List=[
    # #     ##############以下是真实数据###
    #     path_root+ "/orig.png",
    #     path_root+"/HSCNN_plus.png",
    #     path_root+"/CNN3d.png",
    #     path_root+"/MST_Plus_Plus.png",
    #     path_root+"/MIRNet.png",
    #     path_root+"/MPRNet.png",
    #     path_root+"/Restormer.png",
    #     path_root+"/HINet.png",
    #     path_root+"/AWAN.png",
    #     path_root+"/hdnet.png",
    #     path_root+"/our.png",

        ############以下是仿真数据###########
         # path_root + "/hscnn.png",
        # path_root + "/CNN3d.png",
        # path_root + "/MST_Plus_Plus.png",
        # path_root + "/MIRNet.png",
        # path_root + "/MPRNet.png",
        # path_root + "/Restormer.png",
        # path_root + "/HINet.png",
        # path_root + "/AWAN.png",
        # path_root + "/hdnet.png",
        # path_root + "/our.png",
    # ]

    METHOD = [
        # "orig",#####################//zoom 放大的时候加入
        'HSCNN_plus',
        'CNN3d',
        "MST_plus_plus",
        "MIRNet",
        "MPRNet",
        "Restormer",
        "HINet",
        "AWAN",
        'hdnet',
        "our",
#####################################
        # "HSCNN_Plus",  #####################//zoom 2022 数据集
        # '3DCNN',
        # 'MST_pp',
        # "MIRNet",
        # "MPRNet",
        # "Restormer",
        # "HINet",
        # "AWAN",
        # "HDNet",
        # "ours"
    ]

    # data=sio.loadmat("./data/ours.mat")
    PATH_LIST = [
        # './Results/result_sim/orig.mat',
        # './orig.mat',
        # "./data/ours.mat"
  ####################仿真数据 #################################################
        # "./data/HSCNN.mat",
        # "./data/3D.mat",
        # "./data/MST_pp.mat",
        # "./data/MIRNet.mat",
        # "./data/MPRNet.mat",
        # "./data/restormer.mat",
        # "./data/HINet.mat",
        # "./data/AWAN.mat",
        # "./data/HDNet.mat",
        # './data/ours.mat',
    ###########################
        # "./scene03_out/orig.mat",
        # "./scene05_out/hscnn.mat",
        # "./scene05_out/cnn3d.mat",
        # "./scene05_out/MST.mat",
        # "./scene05_out/mirnet.mat",
        # "./scene05_out/mprnet.mat",
        # "./scene05_out/restormer.mat",
        # "./scene05_out/hinet.mat",
        # "./scene05_out/awan.mat",
        # "./scene05_out/hdnet.mat",
        # './scene05_out/ours.mat',
    ############################
#         ''2022 索引
        #################2022 另一个数据点#########
        #         "data4./hscnn_plus.mat",
        #
        #         "data4/CNN3d.mat",
        #         "data4/MST_PP.mat",
        #         "data4/mirnet.mat",
        #         "data4/mprnet.mat",
        #         "data4/restormer.mat",
        # "data4/ours.mat",
        #
        #         "data4/awan.mat",
        #         'data4/hdnet.mat',
        # "data4/hinet.mat",
        ##################################
        # './data1/hdnet.mat',
        # "./data1/hscnn_plus.mat",
        # "./data1/cnn3d.mat",
        # "./data1/mst_pp.mat",
        # "./data1/mirnet.mat",
        # "./data1/mprnet.mat",
        # "./data1/restormer.mat",
        # "./data1/hinet.mat",
        # "./data1/awan.mat",
        # "./data1/ours.mat",
#################real 2022 machines###############################
        # #
        # path_root+"/hscnn_plus.mat",
        # path_root+"/CNN3d.mat",
        # path_root+"/mst_pp.mat",
        # path_root+"/mirnet.mat",
        # path_root+"/mprnet.mat",
        # path_root+"/restormer.mat",
        # path_root+"/hinet.mat",
        # path_root+"/awan.mat",
        # path_root+"/hdnet.mat",
        # path_root+"/ours.mat",

        #############仿真数据#################
        path_root+"/hscnn.mat",
        path_root+"/cnn3d.mat",
        path_root+"/MST.mat",
        path_root+"/mirnet.mat",
        path_root+"/mprnet.mat",
        path_root+"/restormer.mat",
        path_root+"/hinet.mat",
        path_root+"/awan.mat",
        path_root+"/hdnet.mat",
        path_root+"/ours.mat",


        ###################

        # GeneRGB()
################下面是机器人数据############################################
        # "./hscnn_plus.mat",
        # "./CNN3d.mat",
        # "./mst_pp.mat",
        # "./mirnet.mat",
        # "./mprnet.mat",
        # "./restormer.mat",
        # "./hinet.mat",
        # "./awan.mat",
        # "./hdnet.mat",
        # './ours.mat',

#############################
#         # './Results/result_sim/DeSCI.mat',
#         # './Results/result_sim/l-Net.mat',
#
#         # './Results/result_real/Ours/model_225/Ours.mat'
    ]
    # showAllBands()
#     # ref = sio.loadmat('./orig.mat')
#     # print(0)
    local(METHOD, PATH_LIST, 2)#1
#     rgb_local(METHOD,GRB_List,2)

#     showPSNR(METHOD,PATH_LIST)
#     res = sio.loadmat("./data1/cnn3d.mat")['msi']
#     plt.imshow(res[:,:,12],cmap="gray")
#     plt.show()
    cruve(METHOD,PATH_LIST, 1)#2
#     cal_PSRN_RGB(GRB_List,METHOD)
#     cruze_RGB(GRB_List,METHOD)
#     GeneRGB(METHOD,PATH_LIST)
#     # localreal(METHOD,PATH_LIST, 5)
# # 15:59
# # 2022/11/25
# # Administrator
