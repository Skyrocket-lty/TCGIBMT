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
    muzero = 15 #Dataset 2:r=1----Dataset 3:r=15--Dataset 4:r=30-Dataset 5:r=100--Dataset 6:r=35
    mu = muzero
    gamma = 0.06#Dataset 2:gama=10----Dataset 3:gama=30--Dataset 4:gama=3--Dataset 5:gama=50--Dataset 6:gama=0.06
    rho = 2 #Dataset 2:rho=2----Dataset 3:rho=20---Dataset 4:rho=15---Dataset 5:rho=100---Dataset 6:rho=2
    tol = 1e-3
    alpha = 2 #Dataset 2:alpha=2----Dataset 3:alpha=15--Dataset 4:alpha=5--Dataset 5:alpha=100--Dataset 6:alpha=2


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
    for i in range(0,2):#Dataset 2:=1#Dataset 3:=5#Dataset 4:=1#Dataset 5:=5，Dataset 6:=2
        U, S, V = np.linalg.svd(H0)
        r = 20# Dataset 2：r=1;Dataset 3：r=15;Dataset 4：r=11;Dataset 5：r=20;Dataset 6：r=1;
        A = U[:, :r]
        B = V[:r, :]
        H0 = GAMA(H0,A,B)
    Smmi = H0
    return Smmi


