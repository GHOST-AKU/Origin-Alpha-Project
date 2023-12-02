import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io

mat = scipy.io.loadmat('C:\\Users\\GHOST_AKU\\Desktop\\ex.mat')
print(type(mat))
print(mat['X'][0])
print(mat['y'].T[0])

import numpy as np
rows = 10
cols = 10
fig = plt.figure(figsize=(5, 5))
indexes = np.random.choice(5000, rows * cols)
indexes = [i for i in indexes]
#print(len(indexes))

count = 0
for i in range(0, rows):
    for j in range(0, cols):
        axl = fig.add_subplot(rows, cols, count + 1)
        axl.imshow(mat['X'][indexes[count]].reshape(20, 20).T, cmap='gray')  # Remove the space in 'gray'
        axl.autoscale(False)
        axl.set_axis_off()
        count += 1

fig.suptitle("212017074 Xie Zhihao", color='red')
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)  # Replace 'l' with an appropriate value
plt.show()

from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(3,), activation='relu')
nn.fit(mat['X'], mat['y'].T[0])
#print(nn.score(mat['X'], mat['y'].T[0]))
#print(nn.predict(mat['X'][0]. reshape (1,-1)))

from PIL import Image
import numpy as np

# 打开图片
img = Image.open('C:\\Users\\GHOST_AKU\\Desktop\\1.bmp')
# 将图片转换为ndarray
a=np.array(img,dtype=np.uint8)
print(nn.predict(a.reshape(1,-1)))