import matplotlib.pyplot as plt
import numpy as np

x=range(20)

SNR_1=[22.9744, 23.7842, 24.3519, 24.7036, 24.8693, 24.8846, 24.8106, 24.6748, 24.5155, 24.3305, 24.1589, 23.9823, 23.8176, 23.6715, 23.5321, 23.4121, 23.2994, 23.2082, 23.1231, 23.0515]
SNR_5=[21.2612, 22.4585, 23.5754, 24.5853, 25.4229, 26.0537, 26.4662, 26.6827, 26.7468, 26.6932, 26.5741, 26.4034, 26.2303, 26.046, 25.871, 25.7012, 25.5503, 25.4103, 25.2896, 25.1865]
SNR_10=[20.0942, 21.1459, 22.1944, 23.2378, 24.2147, 25.1408, 25.9794, 26.7015, 27.3042, 27.7954, 28.171, 28.436, 28.6195, 28.7386, 28.8029, 28.827, 28.828, 28.8124, 28.7844, 28.7506]
SNR_15=[18.3487, 19.2048, 20.0677, 20.9538, 21.8444, 22.7212, 23.5709, 24.3826, 25.1656, 25.8644, 26.5243, 27.1014, 27.6169, 28.0562, 28.4359, 28.7635, 29.0267, 29.2393, 29.4132, 29.5599]
SNR_19=[17.8751, 18.7078, 19.5342, 20.3828, 21.2526, 22.1145, 22.9599, 23.7823, 24.5929, 25.3412, 26.0631, 26.7292, 27.3258, 27.871, 28.3481, 28.7776, 29.1462, 29.4452, 29.7036, 29.9342] 
##Our try new:
SNR_1=[21.5906, 22.3035, 22.8427, 23.2445, 23.5044, 23.6558, 23.7294, 23.7329, 23.7016, 23.6446, 23.571, 23.4987, 23.4189, 23.3492, 23.2707, 23.2045, 23.1385, 23.0859, 23.0403, 22.9973]
SNR_5=[20.4812, 21.4908, 22.4364, 23.2763, 23.9956, 24.5606, 25.0147, 25.3457, 25.5713, 25.7167, 25.7918, 25.8272, 25.8325, 25.803, 25.759, 25.7164, 25.6675, 25.6117, 25.568, 25.5241]
SNR_10=[19.0194, 19.9715, 20.9348, 21.8802, 22.8056, 23.6408, 24.4504, 25.185, 25.8393, 26.4043, 26.884, 27.2878, 27.6097, 27.8592, 28.0601, 28.2098, 28.325, 28.4098, 28.4666, 28.5114]
SNR_15=[18.1451, 19.017, 19.8972, 20.806, 21.7262, 22.6002, 23.4532, 24.2742, 25.0426, 25.7434, 26.3988, 26.9664, 27.4783, 27.9151, 28.3043, 28.6241, 28.8872, 29.1185, 29.2975, 29.4456]
#SNR_19=

SNR_attention=[22.6201, 23.3544, 24.0489, 24.7094, 25.3604, 25.9559, 26.5362, 27.0705, 27.5642, 28.0014, 28.3856, 28.7117, 28.9853, 29.1971, 29.3773, 29.5183, 29.6338, 29.7216, 29.7954, 29.8513]

#SNR_attention=[22.2909, 23.0134, 23.7117, 24.3719, 25.0017, 25.5896, 26.1592, 26.6707, 27.1291, 27.548, 27.9193, 28.2438, 28.5114, 28.7065, 28.8541, 28.9696, 29.0666, 29.1427, 29.2052, 29.2631]

plt.title('$R=1/6$')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('PSNR (dB)', size=10)
plt.plot(x, SNR_1, color='b', linestyle='-', marker='o',label='EXPLICIT JSCC-OFDM ($SNR_{train}=1dB$)')
plt.plot(x, SNR_5, color='black', linestyle='-', marker='o', label='EXPLICIT JSCC-OFDM ($SNR_{train}=5dB$)')
plt.plot(x, SNR_10, color='g', linestyle='-', marker='o',label='EXPLICIT JSCC-OFDM ($SNR_{train}=10dB$)')
plt.plot(x, SNR_15, color='pink', linestyle='-', marker='o',label='EXPLICIT JSCC-OFDM ($SNR_{train}=15dB$)')
plt.plot(x, SNR_19, color='grey', linestyle='-', marker='o',label='EXPLICIT JSCC-OFDM ($SNR_{train}=19dB$)')
plt.plot(x, SNR_attention, color='r', linestyle='-', marker='*',label='Channel-Adaptive JSCC')
#plt.ylim([18, 32])


plt.legend()
plt.grid()
plt.show()
plt.savefig('./PSNR_OFDM_attention_rate_6.jpg')

