import scipy
import torch
import os
from torch.utils.data import DataLoader

from self_option import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from self_utils import *
from LRTDUNet import *
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
###########SSR_net       RGB---HSI  ---SSRl######################################333
opt.batch_size=6
opt.end_epoch=400
PATH6='/data1/lxx_data/SR_model/'
opt.outf='/data1/lxx_data/DDformer2/'#output path
# opt.outf='/data1/lxx_data/DGUNet/'  #no waveloss
# model=hyf(3,31).cuda().float()
# model=DST().cuda().float()
# model=HYT_Plus_Plus(3,31,31,2).cuda().float()
# checkpoint = torch.load("/data1/lxx_data/DDformer/net_107_epoch_psrn34.9058.pth")
model=MGUHST().cuda().float()
# opt.outf='/data1/lxx_data/DGUNet/net_195_epoch_psrn34.5876.pth'
# checkpoint = torch.load('/data1/lxx_data/DGUNet/net_95_epoch_psrn34.5009.pth')
# model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},strict=True)


per_epoch_iteration = 1000  #epoch iteration settings
total_iteration = per_epoch_iteration*opt.end_epoch
print("\nloading dataset ...")
#bgrs[3,482,512]    hypers:[31,482,512]
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.crop_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))
# model=NAFU_Plus_Plus().cuda().float()
# model=Stoformer().cuda().float()
# model=DST_plus().cuda().float()
##################################
# model=DGUNet().cuda().float()
# model=Our_2mst(3,31,31,2).cuda().float()
# model=MPR_Plus_Plus().cuda().float()
# model=Our
# model.load_state_dict(torch.load(opt.outf))
# model=HINet(31,31,31,4).cuda().float()
# model=AWAN(3,31,32,4).cuda().float()
# model=SSRformer2Net(31).cuda().float()
# model.load_state_dict(torch.load(PATH6))
############################################33
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_ssim=Loss_SSIM()
criterion_wave=Loss_Wavelet()
criterion_spectral=Loss_Spectral()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)
def main():
    iteration = 0
    # record_mrae_loss = 1000
    # total_iteration=1200
    while iteration<total_iteration:
        # list_weight=[0.0001,0.0001,0.0005,0.1,0.9]
        list_weight = [0.0001, 0.0002,0.0005, 0.9]
    # while(True):

        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            # loss = torch.sum([criterion_mrae(output[j], labels) for j in range(len(output))])
            # loss=criterion_mrae(torch.clamp(output[2],0,1),labels)
            # loss=criterion_mrae(output,labels)
            # loss=criterion_wave(output,labels)
            loss= sum([criterion_wave(output[j],labels)*list_weight[j] for j in range(len(output))])+ \
                  criterion_mrae(output[-1], labels)
            # loss=torch.sqrt(criterion_mrae(output[0],labels))+torch.sqrt(criterion_mrae(output))
            # loss =criterion_mrae(output, labels)+criterion_wave(output,labels)
            # loss=criterion_wave(output,labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration=i+1
            if iteration % 20 == 0:
                 print('algorithm:real_mul_wave  BMSY_loss: [iter:%d/%d],lr=%.9f,train_losses.avg=%.9f' % (iteration, total_iteration, lr, losses.avg))
            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss,ssim_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss},SSIM:{ssim_loss}')
                # Save model
                # if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                #     print(f'Saving to {opt.outf}')
                #     save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                #     if mrae_loss < record_mrae_loss:
                #         record_mrae_loss = mrae_loss
                # print loss
                if torch.abs(psnr_loss)>34.40:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf,(iteration//1000),psnr_loss,iteration,model,optimizer)
                ################################
                ###################################
                #########################################
                print(" Epoch[%06d],Iter[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f  Test SSIM: %.9f" % (iteration,iteration/1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss,ssim_loss))
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ssim=AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            output=output[-1]
            # output=output[:,:,:482,:]
            # print(output.shape)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_ssim=criterion_ssim(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_ssim.update(loss_ssim.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg,loss_ssim


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()
    # torch.save(model.state_dict(), PATH6)
    print(torch.__version__)