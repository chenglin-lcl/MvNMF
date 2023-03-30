import numpy as np
from sklearn.cluster import KMeans


class NMF:
    def __init__(self, X, dim, max_iter, init_mth):
        self.X = X  # m*n
        self.dim = dim
        self.max_iter = max_iter
        self.m, self.n = X.shape
        self.init_mth = init_mth

    def normlize_data(self):
        Q = np.diag(1 / np.sqrt(np.sum(self.X ** 2, axis=0)))
        self.X = self.X @ Q

    def __init_U_V(self):
        if self.init_mth == True:
            y_pred = KMeans(n_clusters=self.dim).fit_predict(self.X.transpose())  # n*m
            V = np.zeros([self.dim, self.n], dtype=np.float64)
            for i in range(len(y_pred)):
                V[y_pred[i] - 1, i] = 1
            V = np.maximum(V, 0.01)
            U = self.X @ V.transpose()
        else:
            U = np.abs(np.random.rand(self.m, self.dim))
            V = np.abs(np.random.rand(self.dim, self.n))
        return U, V

    def __update_U(self, U, V):
        U = U * ((self.X @ V.transpose()) / np.maximum(U @ V @ V.transpose(), 1e-9))
        return U

    def __update_V(self, U, V):
        V = V * ((U.transpose() @ self.X) / np.maximum(U.transpose() @ U @ V, 1e-9))
        return V

    def __calc_obj(self, U, V):
        obj = (np.linalg.norm(self.X - U @ V, ord='fro')) ** 2
        return obj

    def update_NMF(self):
        obj = []
        err_cnt = 0
        U, V = self.__init_U_V()
        for iter in range(self.max_iter):
            V = self.__update_V(U, V)
            U = self.__update_U(U, V)
            obj.append(self.__calc_obj(U, V))
            print('iter = {}, obj = {}'.format(iter + 1, obj[iter]))
            if iter > 0 and obj[iter] > obj[iter - 1]:
                err_cnt += 1
        return V, obj, err_cnt