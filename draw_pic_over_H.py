import matplotlib.pyplot as plt
import numpy as np

x=range(20)

noisy_train_noisy_test=[22.2923, 23.0057, 23.7198, 24.3657, 25.0008, 25.5877, 26.1604, 26.6704, 27.1317, 27.54, 27.9179, 28.2476, 28.512, 28.7066, 28.8537, 28.971, 29.0664, 29.1425, 29.2104, 29.2628]
noisy_train_perfect_test=[22.8189, 23.3591, 23.9333, 24.5551, 25.2008, 25.8436, 26.4643, 27.0345, 27.5166, 27.9185, 28.2813, 28.6024, 28.8393, 28.9844, 29.0721, 29.1457, 29.2043, 29.2516, 29.2965, 29.3358]
perfect_train_noisy_test=[19.4549, 20.2902, 21.1857, 22.0855, 22.9609, 23.8446, 24.708, 25.5183, 26.253, 26.868, 27.4583, 27.9524, 28.3691, 28.6998, 28.9672, 29.1819, 29.3623, 29.4845, 29.5947, 29.6824]
perfect_train_perfect_test=[25.4855, 26.007, 26.4988, 26.9677, 27.3941, 27.7683, 28.1153, 28.4172, 28.6853, 28.9146, 29.1106, 29.257, 29.3887, 29.4911, 29.5839, 29.657, 29.7161, 29.7698, 29.8116, 29.8464]


'''
noisy_train_noisy_test=[22.6198, 23.353, 24.0474, 24.7124, 25.3551, 25.9498, 26.5357, 27.0714, 27.5692, 28.0015, 28.3877, 28.7129, 28.9825, 29.2001, 29.3768, 29.5194, 29.6308, 29.7232, 29.7952, 29.8521]
noisy_train_perfect_test=[23.438, 24.0197, 24.6413, 25.3113, 26.017, 26.6979, 27.3381, 27.9033, 28.3713, 28.7373, 29.0427, 29.288, 29.4696, 29.5881, 29.6804, 29.7565, 29.8212, 29.8718, 29.9146, 29.944]
perfect_train_noisy_test=[19.7634, 20.5907, 21.4661, 22.3372, 23.1914, 24.0671, 24.924, 25.7457, 26.4835, 27.1436, 27.7594, 28.2799, 28.7545, 29.1263, 29.4315, 29.675, 29.8701, 30.0155, 30.1337, 30.2207]
perfect_train_perfect_test=[25.6866, 26.2297, 26.751, 27.2261, 27.6743, 28.0763, 28.4425, 28.7703, 29.053, 29.2891, 29.5002, 29.6739, 29.8231, 29.9436, 30.0448, 30.1252, 30.1955, 30.2449, 30.2864, 30.3236]
'''
plt.title('Robustness of Channel-Apative JSCC to the H')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('PSNR (dB)', size=10)
plt.plot(x, noisy_train_perfect_test, color='g', linestyle='-', marker='*',label='trained with $H_{est}$ and tested with $H_{per}$')
plt.plot(x, noisy_train_noisy_test, color='b', linestyle='-', marker='*',label='trained with $H_{est}$ and tested with $H_{est}$')
plt.plot(x, perfect_train_perfect_test, color='r', linestyle='-', marker='*',label='trained with $H_{per}$ and tested with $H_{per}$')
plt.plot(x, perfect_train_noisy_test, color='grey', linestyle='-', marker='o', label='trained with $H_{per}$ and tested with $H_{est}$')


plt.legend()
#plt.show()
plt.savefig('./PSNR_H_or.jpg')

