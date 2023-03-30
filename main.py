import numpy as np
from sklearn.cluster import KMeans
from multi_view_clustering import MvNMF
from mat4py import loadmat
import matplotlib.pyplot as plt
from clustering_metrics import calc_ACC, calc_NMI, calc_Purity

# 读取数据集
file_name = 'ORL_mtv.mat'
data = loadmat(file_name)
# 查看data中的键
print(list(data.keys()))
# 将数据转化为np.array格式
view_num = 3
X_data = np.empty(view_num, dtype=object)
for view_idx in range(view_num):
    X_data[view_idx] = np.squeeze(np.array(data['X'][view_idx], dtype=np.float64)).transpose() # mxn
    print('feature dimension of ' + str(view_idx+1) + '-th view =', X_data[view_idx].shape[0])

Y = np.squeeze(np.array(data['Y'], dtype=np.float64)).transpose() # n*1
print('the number of samples = ', Y.shape)

class_num = len(set(Y))
print('class_num = ', class_num)

# 创建MvNMF对象
model = MvNMF(X_data, dim=class_num, max_iter=500, init_mth=False)
model.normlize_data() # 数据标准化

# 定义聚类指标
ACC = []
NMI = []
Purity = []
err_cnt = []
obj = []

# 迭代更新
for i in range(10):
    V, obj_val, cnt= model.update_MvNMF()
    # 计算聚类指标
    y_pred = KMeans(n_clusters=class_num).fit_predict(V.transpose())+1
    err_cnt.append(cnt)
    obj.append(obj_val)
    ACC.append(calc_ACC(Y, y_pred))
    NMI.append(calc_NMI(Y, y_pred))
    Purity.append(calc_Purity(Y, y_pred))

# 打印结果
print('err_cnt = ', err_cnt[0])
print('ACC = {:.2f} + {:.2f}'.format(np.mean(ACC)*100, np.std(ACC)*100))
print('NMI = {:.2f} + {:.2f}'.format(np.mean(NMI)*100, np.std(NMI)*100))
print('Purity = {:.2f} + {:.2f}'.format(np.mean(Purity)*100, np.std(Purity)*100))

plt.plot(obj[0])
plt.show()




