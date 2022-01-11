#python train.py --tran_know_flag 1 --model DAS_JSCC_OFDM --all_epoch 200 
#python train.py --tran_know_flag 0 --train_snr 15 
#python train.py --tran_know_flag 0 --train_snr 19 
#nohup bash traun_task.sh > nohup_attention_rate_16.out 2>&1

#python train.py --tran_know_flag 1 --model DAS_JSCC_OFDM --all_epoch 200 --resume True;
#python eval.py --tran_know_flag 1 --model DAS_JSCC_OFDM
python train.py --tran_know_flag 0 --train_snr 1;
nvidia-smi;
python train.py --tran_know_flag 0 --train_snr 5;
nvidia-smi;
python train.py --tran_know_flag 0 --train_snr 10;
nvidia-smi;
python train.py --tran_know_flag 0 --train_snr 15;
nvidia-smi;
python train.py --tran_know_flag 0 --train_snr 19;
nvidia-smi;

python eval.py --tran_know_flag 0 --model JSCC_OFDM --train_snr 1;
python eval.py --tran_know_flag 0 --model JSCC_OFDM --train_snr 5;
python eval.py --tran_know_flag 0 --model JSCC_OFDM --train_snr 10;
python eval.py --tran_know_flag 0 --model JSCC_OFDM --train_snr 15;
python eval.py --tran_know_flag 0 --model JSCC_OFDM --train_snr 19;

