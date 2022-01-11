import matplotlib.pyplot as plt
import numpy as np

x=range(20)

trans_know=[27.66267, 28.35991, 28.99682, 29.59741, 30.13229, 30.63343, 31.09321, 31.49937, 31.85631, 32.19445, 32.49512, 32.76718, 33.00796, 33.21632, 33.39224, 33.55262, 33.6863, 33.79335, 33.88524, 33.96239]

trans_unknow=[27.46611, 28.18551, 28.82179, 29.42081, 29.93794, 30.43671, 30.88031, 31.27268, 31.63954, 31.94964, 32.2377, 32.47688, 32.70282, 32.88987, 33.06711, 33.20527, 33.33293, 33.43358, 33.5155, 33.58102]

plt.title('Performance of different models')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('Ave_PSNR (dB)', size=10)
plt.plot(x, trans_know, color='black', linestyle='-', marker='o', label='trans_know')
plt.plot(x, trans_unknow, color='g', linestyle='-', marker='*',label='trans_unknow')

plt.legend()
plt.show()
plt.savefig('./PSNR_OFDM_trans.jpg')

