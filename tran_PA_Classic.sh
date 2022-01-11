
python train.py --model JSCC_OFDM --tran_know_flag 0 --train_snr 10 --hard_PA 1;
nvidia-smi;
python eval.py --tran_know_flag 0 --model JSCC_OFDM --train_snr 10 --hard_PA 1;

python train.py --model JSCC_OFDM --tran_know_flag 0 --train_snr 5 --hard_PA 1;
nvidia-smi;
python eval.py --tran_know_flag 0 --model JSCC_OFDM --train_snr 5 --hard_PA 1;


python train.py --model JSCC_OFDM --tran_know_flag 0 --train_snr 15 --hard_PA 1;
nvidia-smi;
python eval.py --tran_know_flag 0 --model JSCC_OFDM --train_snr 15 --hard_PA 1;

#nohup bash traun_task.sh > nohup_attention_rate_16.out 2>&1