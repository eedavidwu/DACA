import matplotlib.pyplot as plt
import numpy as np

x=[1,5,10,15]

SNR_JSCC=[22.30,24.57,26.88378,28.62652,]

SNR_attention_h=[23.37,25.64,28.09,29.92988,]


plt.xlabel('$SNR_{train}$ (dB)')
plt.ylabel('PSNR (dB) ($SNR_{test}=SNR_{train}$)')
#plt.plot(x, SNR_1, color='b', linestyle='-', marker='o',label='JSCC-OFDM ($SNR_{train}=1dB$)')
plt.plot(x, SNR_JSCC, color='black', linestyle='-', marker='o', label='EXPLICIT JSCC-OFDM')
#plt.plot(x, SNR_10, color='g', linestyle='-', marker='o',label='JSCC-OFDM ($SNR_{train}=10dB$)')
#plt.plot(x, SNR_15, color='pink', linestyle='-', marker='o',label='JSCC-OFDM ($SNR_{train}=15dB$)')
#plt.plot(x, SNR_19, color='grey', linestyle='-', marker='o',label='JSCC-OFDM ($SNR_{train}=19dB$)')
plt.plot(x, SNR_attention_h, color='r', linestyle='-', marker='*',label='Channel-Adaptive JSCC')
#plt.ylim([18, 32])


plt.legend()
plt.show()
plt.grid()
plt.savefig('./PSNR_OFDM_H_adapts.jpg')

