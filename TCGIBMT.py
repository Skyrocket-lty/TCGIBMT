# -*- codeing = utf-8 -*-
# @Time : 2023/3/11 16:20
# @Author : Chuanru Ren
# @File : RPCAΓNR.py
# @Software: PyCharm

import numpy as np

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy
import Global_similarity
import Local_similarity
import DGI

CC = np.loadtxt(r"circRNA_SMF.txt", dtype=float)
DD = np.loadtxt(r"disease_SMF.txt", dtype=float)
Y= np.loadtxt(r"association.txt",dtype=float)
circRNA_disease_k = np.loadtxt(r"known.txt",dtype=int)
circRNA_disease_uk = np.loadtxt(r"unknown.txt",dtype=int)


def DC(D,mu,T0,g):
    U,S,V = np.linalg.svd(D)
    T1 = np.zeros(np.size(T0))
    for i in range(1,100):
        T1 = DCInner(S,mu,T0,g)
        err = np.sum(np.square(T1-T0))
        if err < 1e-6:
            break
        T0 = T1


    #求行块结果
    V = V[:39, :]
    l_1 = np.dot(U, np.diag(T1))
    l = np.dot(l_1, V)
    return l,T1


def DCInner(S,mu,T_k,gam):
    lamb = 1/mu
    grad = (1+gam)*gam/(np.square(gam+T_k))
    T_k1 = S-lamb*grad
    T_k1[T_k1<0]=0
    return T_k1

#上面是求奇异值的，下面是求L2，1范数的

def GAMA(H,A,B):
    muzero = 15 #Dataset 2:r=1----Dataset 3:r=15--Dataset 4:r=30-Dataset 5:r=100
    mu = muzero
    gamma = 0.06#Dataset 2:gama=10----Dataset 3:gama=30--Dataset 4:gama=3--Dataset 5:gama=50
    rho = 2 #Dataset 2:rho=2----Dataset 3:rho=20---Dataset 4:rho=15---Dataset 5:rho=100
    tol = 1e-3
    alpha = 2 #Dataset 2:alpha=2----Dataset 3:alpha=15--Dataset 4:alpha=5--Dataset 5:alpha=100


    m, n = np.shape(H)
    L = copy.deepcopy(H)
    S = np.zeros((m,n))
    Y = np.zeros((m,n))  #这个保存，正常更新

    omega = np.zeros(H.shape)
    omega[H.nonzero()] = 1

    for i in range(0, 500):
        #这些代码是求W的
        tran = (1/mu) * (Y+alpha*(H*omega)+np.dot(A,B))+L
        W = tran - (alpha/(alpha+mu))*omega*tran
        W[W < 0] = 0
        W[W > 1] = 1

        #这三项整体算是求奇异值的,也就是X,在这里L就相当于X了
        D = W-Y/mu  #更新C
        sig = np.zeros(min(m, n)) #存奇异值的
        L, sig = DC(copy.deepcopy(D),mu,copy.deepcopy(sig),gamma) #求奇异值的

        #求Y
        Y= Y+mu*(L-W)     #更新Y
        mu = mu*rho         #更新u
        sigma = np.linalg.norm(L-W,'fro')
        RRE = sigma/np.linalg.norm(H,'fro')
        if RRE < tol:
            break
    return W



def truncated(H0):
    for i in range(0,2):#Dataset 2:=1#Dataset 3:=5#Dataset 4:=1#Dataset 5:=5
        U, S, V = np.linalg.svd(H0)
        r = 20# Dataset 2：r=1;Dataset 3：r=15;Dataset 4：r=11;Dataset 5：r=20;
        A = U[:, :r]
        B = V[:r, :]
        H0 = GAMA(H0,A,B)
    Smmi = H0
    return Smmi


def main():
    roc_sum, time = 0, 0
    kf = KFold(n_splits=5, shuffle=True, random_state=9999)#//初始化kfold
    for train_index,test_index in kf.split(circRNA_disease_k):
        X_2 = copy.deepcopy(Y)

        for index in test_index:

            X_2[circRNA_disease_k[index, 0], circRNA_disease_k[index, 1]] = 0

        G_circRNA = Global_similarity.compute_global_similarity_matrix(CC)

        # 计算disease的全局相似性

        G_disease = Local_similarity.compute_global_similarity_matrix(DD)
        #
        #
        Yh1 = DGI.fHGI(0.1, G_circRNA, G_disease, X_2)#Dataset 2为0.1；#Dataset 3为0.1；#Dataset 4为0.09#Dataset 5为0.2；
        # # #计算局部图推理
        # # # 计算circRNA的局部相似性
        L_circRNA = Global_similarity.row_normalization(CC, 5)#Dataset 2为10#Dataset 3为100#Dataset 4为20#Dataset 5为8
        # # 计算disease的局部相似性
        L_disease = Local_similarity.row_normalization(DD, 5)#Dataset 2为5#Dataset 3为10#Dataset 4为10#Dataset 5为8
        Yh2 = DGI.fHGI(0.1, L_circRNA, L_disease, X_2)#Dataset 2为0.1#Dataset 3为0.1#Dataset 4为0.1#Dataset 5为0.2
        L_1 = (Yh1 + Yh2)/2
        H = np.hstack((CC, L_1))  # 将参数元组的元素数组按水平方向进行叠加
        M_1 = truncated(H)
        #行块
        M_1 = M_1[0:CC.shape[0], CC.shape[0]:H.shape[1]]  # 把补充的关联矩阵原来A位置给取出来。
        Label = np.zeros(circRNA_disease_uk.shape[0] + test_index.size)
        Score = np.zeros(circRNA_disease_uk.shape[0] + test_index.size)
        i , j = 0 , 0
        for s_index in test_index:
            Label[i] = 1
            Score[i] = M_1[circRNA_disease_k[s_index,0],circRNA_disease_k[s_index,1]]
            i = i + 1

        for i in range(test_index.size, circRNA_disease_uk.shape[0] + test_index.size):
            Score[i] = M_1[circRNA_disease_uk[j,0],circRNA_disease_uk[j,1]]
            j = j + 1
        fpr, tpr, thersholds = roc_curve(y_true=Label, y_score=Score, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        roc_sum = roc_sum + roc_auc
        time += 1
        s=roc_sum/time
        print(time,roc_auc,roc_sum,s)
        while(time==5):
            return s


if __name__ == "__main__":
    total,time=0,0
    for i in range(0,1):
        l=main()
        print("\n")
        total=total+l
        time +=1
        print(time,total/time)
        print("\n")
