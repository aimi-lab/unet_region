from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


root = '/home/laurent.lejeune/medical-labeling'
path = pjoin(root, 'unet_region', 'runs', '2019-04-10_11-38-11')

df = pd.read_csv(pjoin(path, 'scores.csv'))
thr = df['Unnamed: 0']
s = df['score']
i_max = np.argmax(s)

plt.plot(thr, s, 'b')
plt.plot(thr[i_max], s[i_max], 'bo')
plt.grid()
plt.title('validation set. thr_opt: {}'.format(thr[i_max]))
plt.xlabel('threshold')
plt.ylabel('mean F1 score')
plt.savefig(pjoin(path, 'thr.png'))
plt.show()
