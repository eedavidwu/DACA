import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import model.OFDM as OFDM_models
import os
import argparse
import cv2
from random import randint
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

          
                 
def compute_AvePSNR(model,dataloader,snr):
    psnr_all_list = []
    model.eval()
    MSE_compute = nn.MSELoss(reduction='none')
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        inputs = Variable(inputs.cuda())
        outputs = model(inputs,snr)
        MSE_each_image = (torch.sum(MSE_compute(outputs, inputs).view(b,-1),dim=1))/(c*h*w)
        PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
        one_batch_PSNR=PSNR_each_image.data.cpu().numpy()
        psnr_all_list.extend(one_batch_PSNR)
    Ave_PSNR=np.mean(psnr_all_list)
    Ave_PSNR=np.around(Ave_PSNR,4)

    return Ave_PSNR


def main():
    parser = argparse.ArgumentParser()
    #Train:
    parser.add_argument("--best_ckpt_path", default='./ckpts/', type=str,help='best model path')
    parser.add_argument("--all_epoch", default='150', type=int,help='Train_epoch')
    parser.add_argument("--best_choice", default='loss', type=str,help='select epoch [loss/PSNR]')
    parser.add_argument("--flag", default='train', type=str,help='train or eval for JSCC')
    parser.add_argument("--attention_num", default=0, type=int,help='attention_number')

    # Model and Channel:
    parser.add_argument("--model", default='DAS_JSCC_OFDM', type=str,help='Model select: DAS_JSCC_OFDM/JSCC_OFDM/DAS_ALL_JSCC_OFDM')
    parser.add_argument("--channel_type", default='awgn', type=str,help='awgn/slow fading/burst')
    #parser.add_argument("--const_snr", default=True,help='SNR (db)')
    #parser.add_argument("--input_const_snr", default=1, type=float,help='SNR (db)')
    parser.add_argument("--cp_num", default='16', type=int,help='CP num, 0.25*subcariier')
    parser.add_argument("--gama", default='4', type=int,help='time delay constant for multipath fading channel')
    #PAPR loss:
    parser.add_argument("--h_stddev", default=1, type=float,help='awgn/slow fading/burst')
    parser.add_argument("--equalization", default=1, type=float,help='Equalization_flag')
    parser.add_argument("--S", default=2, type=int,help='number of symbol')
    parser.add_argument("--M", default=64, type=int,help='number of subcarrier')
    parser.add_argument("--tcn", default=4, type=int,help='tansmit_channel_num for djscc')
    parser.add_argument("--N_pilot", default=1, type=int,help='number of pilot symbol')

    parser.add_argument("--input_snr_max", default=20, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_min", default=0, type=int,help='SNR (db)')
    parser.add_argument("--train_snr_list",nargs='+', type=int, help='Train SNR (db)')
    #parser.add_argument("--train_snr_list_in",nargs='+', type=list, help='Train SNR (db)')
    parser.add_argument("--tran_know_flag", type=int,help='tansmit_know flag')
    parser.add_argument("--hard_PA", default=0, type=int,help='PA is hard or soft')



    parser.add_argument("--train_snr", default=10,type=int, help='Train SNR (db)')

    parser.add_argument("--resume", default=False,type=bool, help='Load past model')
    #parser.add_argument("--snr_num",default=4,type=int,help="num of snr")

    # [symbol,tcn]:
    # 1/12 2,4; 1/6 4,8; 1/3 8,16

    GPU_ids = [0,1,2,3]

    global args
    args=parser.parse_args()

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)
    #model_name='JSCC_OFDM'
    #train_snr=5
    ckpt_folder='./ckpts_4/ckpts'

    if args.model=='DAS_JSCC_OFDM':
        train_snr='random'
        auto_encoder=OFDM_models.Attention_all_JSCC(args)
        model_path=os.path.join(ckpt_folder,'best_fading_transmit_1_4_'+args.model+'_SNR_'+str(train_snr)+'.pth')
        #python eval.py --tran_know_flag 1 --model DAS_JSCC_OFDM

    elif args.model=='JSCC_OFDM':
        train_snr=args.train_snr
        auto_encoder=OFDM_models.Classic_JSCC(args)
        #model_path=os.path.join(ckpt_folder,'best_fading_4_'+args.model+'_SNR_'+str(train_snr)+'.pth')
        model_path='./ckpts_8/best_fading_H_'+args.model+'_SNR_'+str(train_snr)+'.pth'

    #for train_snr in ['5','10','15','19']:
        #train_snr='5'
        #train_snr='random_in_list'
        #Create model
    auto_encoder = nn.DataParallel(auto_encoder,device_ids = GPU_ids)
    auto_encoder = auto_encoder.cuda()
    print("Create the model:",args.model)

    checkpoint=torch.load(model_path)
    epoch_last=checkpoint["epoch"]
    auto_encoder.load_state_dict(checkpoint["net"])

    best_psnr=checkpoint["Ave_PSNR"]
    Trained_SNR=checkpoint['SNR']

    print("Load model:",model_path)
    print("Model is trained in SNR: ",train_snr," with PSNR:",best_psnr," at epoch ",epoch_last)

    #PSNR_list=[[0 for a in range(6)] for b in range(6)]
    #SNR_1_list=[1,4,9,12,16,19]
    #SNR_2_list=[1,4,9,12,16,19]
    PSNR_list=[]

    #for i in range(20):
    for i in [1,10,19]:

        #i=4
        #j=2
        validate_snr=i
        one_ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
        print("Compute Ave_PSNR in SNR:",validate_snr,"with PSNR:",one_ave_psnr)
        PSNR_list.append(one_ave_psnr)
        #PSNR_list[i]=one_ave_psnr
    print("Finish validate",model_path)  
    print(PSNR_list)  
if __name__ == '__main__':
    main()
