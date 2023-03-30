import numpy as np
from sklearn.cluster import KMeans


class MvNMF:
    def __init__(self, X, dim, max_iter=500, init_mth=True, threshold=1e-4):
        self.X = X  # 多视图m*n
        self.dim = dim # k
        self.view_num = len(self.X)
        self.max_iter = max_iter
        self.init_mth = init_mth # 选择初始化方法
        self.threshold = threshold
        self.m = [self.X[i].shape[0] for i in range(self.view_num)]
        self.n = self.X[0].shape[1]

    def normlize_data(self):
        for view_idx in range(self.view_num):
            Q = np.diag(1 / np.sqrt(np.sum(self.X[view_idx] ** 2, axis=0)))
            self.X[view_idx] = self.X[view_idx] @ Q

    def __init_U_V(self):
        U = np.empty(self.view_num, dtype=object)
        V = np.empty(self.view_num, dtype=object)

        if self.init_mth == True:
            # 利用kmeans进行初始化
            for view_idx in range(self.view_num):
                y_pred = KMeans(n_clusters=self.dim).fit_predict(self.X[view_idx].transpose())  # n*m
                V[view_idx] = np.zeros([self.dim, self.n], dtype=np.float64)
                for i in range(len(y_pred)):
                    V[view_idx][y_pred[i]-1, i] = 1
                V[view_idx] = np.maximum(V[view_idx], 0.01)
                U[view_idx] = self.X[view_idx] @ V[view_idx].transpose()
        else:
            # 随机初始化
            for view_idx in range(self.view_num):
                U[view_idx] = np.abs(np.random.rand(self.m[view_idx], self.dim))
                V[view_idx] = np.abs(np.random.rand(self.dim, self.n))
        return U, V

    def __update_U(self, U, V):
        for view_idx in range(self.view_num):
            temp_up = self.X[view_idx] @ V[view_idx].transpose()
            temp_down = U[view_idx] @ V[view_idx] @ V[view_idx].transpose()
            U[view_idx] = U[view_idx] * (temp_up / np.maximum(temp_down, 1e-9))
        return U

    def __update_V(self, U, V):
        for view_idx in range(self.view_num):
            temp_up = U[view_idx].transpose() @ self.X[view_idx]
            temp_down = U[view_idx].transpose() @ U[view_idx] @ V[view_idx]
            V[view_idx] = V[view_idx] * (temp_up / np.maximum(temp_down, 1e-9))
        return V

    def __calc_obj(self, U, V):
        obj = 0
        for view_idx in range(self.view_num):
            obj += (np.linalg.norm(self.X[view_idx] - U[view_idx] @ V[view_idx], ord='fro')) ** 2
        return obj

    def update_MvNMF(self):
        obj = []
        err_cnt = 0
        U, V = self.__init_U_V()
        for iter in range(self.max_iter):
            V = self.__update_V(U, V)
            U = self.__update_U(U, V)
            obj.append(self.__calc_obj(U, V))
            print('iter = {}, obj = {:.2f}'.format(iter + 1, obj[iter]))
            if iter > 0 and obj[iter] > obj[iter - 1]:
                err_cnt += 1
            if iter > 1 and np.abs(obj[iter] - obj[iter-1])/obj[iter-1] < self.threshold:
                break
        V_star = np.full_like(V[0], 0)
        for view_idx in range(self.view_num):
            V_star += V[view_idx]/self.view_num
        return V_star, obj, err_cnt
