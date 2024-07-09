import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 生成一些示例数据
true_labels = [0, 1, 2, 1, 0, 2]
predicted_labels = [0, 1, 1, 1, 0, 2]

# 计算混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels)

# 绘制混淆矩阵图表
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# 添加文本标签
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white")

# 设置坐标轴标签等
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# 展示图表
plt.show()