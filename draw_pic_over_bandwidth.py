import matplotlib.pyplot as plt
import numpy as np

x=range(20)
SNR_attention_2=[20.59611, 21.20879, 21.78124, 22.32425, 22.86043, 23.35584, 23.81904, 24.24444, 24.63202, 24.97084, 25.27781, 25.52679, 25.73124, 25.9017, 26.0366, 26.13902, 26.22253, 26.2922, 26.35134, 26.39674]
SNR_attention_4=[22.8702, 23.6047, 24.2924, 24.9444, 25.5386, 26.0959, 26.588, 27.0516, 27.4542, 27.824, 28.1564, 28.4575, 28.7109, 28.9403, 29.1349, 29.3071, 29.4528, 29.5715, 29.6755, 29.759]
SNR_attention_8=[24.21228, 25.03544, 25.82678, 26.56092, 27.28298, 27.94467, 28.57928, 29.16796, 29.71776, 30.2278, 30.69616, 31.12796, 31.5279, 31.89536, 32.21882, 32.51213, 32.77456, 32.99307, 33.18502, 33.33776]
#SNR_attention=[33.11577, 33.11031, 33.10403]
plt.title('Performance of channel-adpative JSCC over different R')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('Ave_PSNR (dB)', size=10)
plt.plot(x, SNR_attention_8, color='r', linestyle='-', marker='o', label='R=1/3')
plt.plot(x, SNR_attention_4, color='g', linestyle='-', marker='*',label='R=1/6')
plt.plot(x, SNR_attention_2, color='b', linestyle='-', marker='*',label='R=1/12')
plt.ylim([17, 34])


plt.legend()
#plt.show()
plt.savefig('./PSNR_OFDM_atten_over_ratio.jpg')
print("done_2")
