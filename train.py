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
from random import randint
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def train(args,auto_encoder,trainloader,testloader,train_snr):
    #model_name:
    model_name=args.model
    
    # Define an optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = optim.Adam(auto_encoder.parameters())
    
    #Start Train:
    batch_iter=(trainloader.dataset.__len__() // trainloader.batch_size)
    print_iter=int(batch_iter/2)
    best_psnr=0
    epoch_last=0
    best_auto=auto_encoder

    #whether resume:
    if args.resume==True:
        #model_path=os.path.join(args.best_ckpt_path,'best_fading_rate_8_transmit_'+str(args.tran_know_flag)+'_equal_1_'+model_name+'_SNR_'+str(train_snr)+'.pth')
        model_path=os.path.join('./ckpts','best_fading_rate_4_transmit_1_equal_1_'+args.model+'_SNR_'+str(train_snr)+'.pth')

        #model_path=os.path.join(args.best_ckpt_path,'best_weight_'+model_name+'_SNR_H_'+str(train_snr)+'.pth')
        checkpoint=torch.load(model_path)
        epoch_last=checkpoint["epoch"]
        auto_encoder.load_state_dict(checkpoint["net"])

        optimizer.load_state_dict(checkpoint["op"])
        best_psnr=checkpoint["Ave_PSNR"]
        Trained_SNR=checkpoint['SNR']
        #optimizer=checkpoint['op']

        print("Load model:",model_path)
        print("Model is trained in SNR: ",train_snr," with PSNR:",best_psnr," at epoch ",epoch_last)
        auto_encoder = auto_encoder.cuda()

    for epoch in range(epoch_last,args.all_epoch):
        auto_encoder.train()
        running_loss = 0.0

        channel_snr=train_snr
        channel_flag=train_snr
        #print('Epoch ',str(epoch),' trained with SNR: ',channel_flag)
        
        for batch_idx, (inputs, _) in enumerate(trainloader, 0):
            inputs = Variable(inputs.cuda())
            # set a random noisy:            
            # ============ Forward ============
            #papr,outputs = auto_encoder(inputs,channel_snr)
            papr=0
            outputs = auto_encoder(inputs,channel_snr)
            loss_mse=criterion(outputs, inputs)
            # ============ Backward ============
            optimizer.zero_grad()
            if args.papr_flag==True:
                #papr_db=10*torch.log10(papr)
                loss_ave_papr=papr.mean()
                #loss_ave_papr=papr_db.mean()
                loss = loss_mse+args.papr_lambda*loss_ave_papr
            else:
                loss = loss_mse

            loss.backward()
            optimizer.step()
            # ============ Ave_loss compute ============
            running_loss += loss.data

            if (batch_idx+1) % print_iter == 0:
                if args.papr_flag==True:
                    print("Epoch: [%d] [%4d/%4d] , loss: %.5f, loss_mse: %.5f,  loss_weighted_PAPR: %.5f, loss_PAPR: %.5f" %
                        ((epoch), (batch_idx), (batch_iter),loss,loss_mse ,(args.papr_lambda*loss_ave_papr),loss_ave_papr,))
                else:
                    print("Epoch: [%d] [%4d/%4d] , loss: %.8f" %
                        ((epoch), (batch_idx), (batch_iter), running_loss / print_iter))
                running_loss = 0.0
        
        if (epoch % 4) ==0:
            ##Validate:
            if args.model=='JSCC_OFDM':
                validate_snr=channel_snr
                ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
                if args.papr_flag==True:
                    print("############## Validate model with SNR: ",validate_snr," PAPR lambda: ",str(args.papr_lambda),", and get Ave_PSNR:",ave_psnr," ##############")
                else:
                    print("############## Validate model with SNR: ",validate_snr,", and get Ave_PSNR:",ave_psnr," ##############")

                if ave_psnr > best_psnr:
                    PSNR_list=[]
                    best_psnr=ave_psnr
                    print('Find one best model with PSNR:',best_psnr,' under SNR: ',channel_flag)
                    #for i in [1,4,10,16,19]:
                    '''
                    for i in [1,5,10,15,19]:
                        validate_snr=i
                        one_ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
                        PSNR_list.append(one_ave_psnr)
                    #print("in:[1,4],[9,12],[16,19]")
                    print(PSNR_list)
                    '''
                    checkpoint={
                        "model_name":args.model,
                        "net":auto_encoder.state_dict(),
                        "op":optimizer.state_dict(),
                        "epoch":epoch,
                        "SNR":channel_flag,
                        "Ave_PSNR":ave_psnr
                    }
                    save_path=os.path.join(args.best_ckpt_path,'best_fading_rate_'+str(args.tcn)+'_'+model_name+'_SNR_'+str(channel_snr)+'.pth')
                    best_auto=auto_encoder

                    '''
                    if args.papr_flag==True:
                        if args.clip_flag==True:
                            save_path=os.path.join(args.best_ckpt_path,'best_weight_fading_H_PAPR_with_clip'+model_name+'_SNR_'+str(channel_snr)+'.pth')
                        else:
                            save_path=os.path.join(args.best_ckpt_path,'best_weight_fading_H_PAPR_'+str(args.papr_lambda)+'_'+model_name+'_SNR_'+str(channel_snr)+'.pth')
                    else:
                        if args.clip_flag==True:
                            save_path=os.path.join(args.best_ckpt_path,'best_weight_fading_H_with_clip'+model_name+'_SNR_'+str(channel_snr)+'.pth')
                        else:
                            save_path=os.path.join(args.best_ckpt_path,'best_weight_fading_H_'+model_name+'_SNR_'+str(channel_snr)+'.pth')
                    '''

                    torch.save(checkpoint, save_path)
                    print('Saving Model at epoch',epoch,save_path)
            if args.model=='DAS_JSCC_OFDM':
                if channel_snr=='random':
                    PSNR_list=[]
                    for i in [1,5,10,15,19]:
                            validate_snr=i
                            one_ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
                            PSNR_list.append(one_ave_psnr)
                    #print("in:[1,4],[9,12],[16,19]")
                    ave_psnr=np.mean(PSNR_list)
                else:
                    validate_snr=channel_snr
                    ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
                    PSNR_list=ave_psnr

                #print(PSNR_list)
                
                #ave_psnr=compute_AvePSNR(auto_encoder,testloader,10)
   
                print("############## Validate model with SNR: ",channel_snr,", and get Ave_PSNR:",ave_psnr," ##############")
                
                if ave_psnr > best_psnr:
                    best_psnr=ave_psnr
                    print("### Find one best PSNR List:",PSNR_list,"###")
                    print('Find one best model with PSNR:',best_psnr,' under SNR List [1,5,10,15,19]')

                    checkpoint={
                        "model_name":args.model,
                        "net":auto_encoder.state_dict(),
                        "op":optimizer.state_dict(),
                        "epoch":epoch,
                        "SNR":channel_flag,
                        "Ave_PSNR":ave_psnr
                    }
                    save_path=os.path.join(args.best_ckpt_path,'best_fading_rate_'+str(args.tcn)+'_transmit_'+str(args.tran_know_flag)+'_equal_'+str(args.equalization)+'_'+model_name+'_SNR_'+str(channel_snr)+'.pth')
                    torch.save(checkpoint, save_path)
                    print('Saving Model at epoch',epoch,'at path:',save_path)
          
                 
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
    Ave_PSNR=np.around(Ave_PSNR,5)

    return Ave_PSNR


def main():
    parser = argparse.ArgumentParser()
    #Train:
    parser.add_argument("--best_ckpt_path", default='./ckpts_h_ad/', type=str,help='best model path')
    parser.add_argument("--all_epoch", default=100, type=int,help='Train_epoch')
    parser.add_argument("--best_choice", default='loss', type=str,help='select epoch [loss/PSNR]')
    parser.add_argument("--flag", default='train', type=str,help='train or eval for JSCC')
    parser.add_argument("--attention_num", default=64, type=int,help='attention_number')

    # Model and Channel:
    parser.add_argument("--model", default='DAS_JSCC_OFDM', type=str,help='Model select: DAS_JSCC_OFDM/JSCC_OFDM')
    parser.add_argument("--channel_type", default='awgn', type=str,help='awgn/slow fading/burst')
    parser.add_argument("--h_stddev", default=1, type=float,help='awgn/slow fading/burst')
    parser.add_argument("--equalization", default=1,type=int,help='Equalization_flag 1.eq 2.cat')
    parser.add_argument("--S", default=4, type=int,help='number of symbol')
    parser.add_argument("--M", default=64, type=int,help='number of subcarrier')
    parser.add_argument("--N_pilot", default=1, type=int,help='number of pilot symbol')
    parser.add_argument("--tcn", default=8, type=int,help='tansmit_channel_num for djscc')
    parser.add_argument("--tran_know_flag", default=1, type=int,help='tansmit_know flag')
    parser.add_argument("--H_perfect", default=0, type=int,help='H perfect or not')

    parser.add_argument("--cp_num", default='8', type=int,help='CP num, 0.25*subcariier')
    parser.add_argument("--gama", default='4', type=int,help='time delay constant for multipath fading channel')

    #PAPR loss:
    parser.add_argument("--papr_lambda", default=0.0005,type=float,help='PAPR parameter')
    parser.add_argument("--papr_flag", default=False,type=bool,help='PAPR parameter')
    parser.add_argument("--clip_flag", default=False,type=bool,help='PAPR parameter')


    parser.add_argument("--input_snr_max", default=20, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_min", default=0, type=int,help='SNR (db)')
    parser.add_argument("--train_snr_list",nargs='+', type=int, help='Train SNR (db)')
    #parser.add_argument("--train_snr_list_in",nargs='+', type=list, help='Train SNR (db)')

    parser.add_argument("--train_snr",default=5,type=int, help='Train SNR (db)')

    parser.add_argument("--resume", default=False,type=bool, help='Load past model')
    #parser.add_argument("--snr_num",default=4,type=int,help="num of snr")

    GPU_ids = [0,1,2,3]


    global args
    args=parser.parse_args()

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)
    
    # Create model
    # Create model
    print('equalization:',args.equalization)
    print('h_stdev',args.h_stddev)
    print('trainsmitter know:',args.tran_know_flag)
    print('esitmate h perfect:',args.H_perfect)

    if args.model=='JSCC_OFDM':
        auto_encoder=OFDM_models.Classic_JSCC(args)
        auto_encoder = nn.DataParallel(auto_encoder,device_ids = GPU_ids)
        auto_encoder = auto_encoder.cuda()
        print("Create the model:",args.model)
        #nohup python train.py --train_snr 10 --tran_know_flag 0 --equalization 2 > nohup_unknown_jscc_10.out&  know:26214->unknwo:25833
        train_snr=args.train_snr
        #train_snr=10

        print("############## Train model with SNR: ",train_snr," ##############")
        train(args,auto_encoder,trainloader,testloader,train_snr)
    if args.model=='DAS_JSCC_OFDM':
        auto_encoder=OFDM_models.Attention_all_JSCC(args)
        auto_encoder = nn.DataParallel(auto_encoder,device_ids = GPU_ids)
        auto_encoder = auto_encoder.cuda()
        print("Create the model:",args.model)
        #train_snr=args.train_snr
        train_snr='random'
        #train_snr=args.train_snr
        #nohup python train.py --train_snr_list 11 19 > nohup_11_19.out&
        #nohup python train.py --train_snr 6 > nohup_OFDM_6.out&
        print("############## Train model with SNR: ",train_snr," ##############")
        train(args,auto_encoder,trainloader,testloader,train_snr)
    
if __name__ == '__main__':
    main()
    #nohup bash tran.sh > test.out 2>&1&
    #nohup python train.py --model DAS_JSCC_OFDM --S 2 --tcn 4 > train_rate_4_attention.out&
    #nohup python train.py --model JSCC_OFDM --train_snr 10 --tran_know_flag 0 > nohup_JSCC_10.out

