import numpy as np
import norm
def fHGI(alpha, A_CC, A_DD, A_CD):

    normWCC = norm.normFun(A_CC) #这是circRNA相似性矩阵
    normWDD = norm.normFun(A_DD) #这是疾病相似性矩阵

    Wdr0 = A_CD
    Wdr_i = Wdr0

    Wdr_I = alpha * normWCC @ Wdr_i @ normWDD + (1 - alpha) * Wdr0


    while np.max(np.abs(Wdr_I - Wdr_i)) > 1e-10:
        Wdr_i = Wdr_I
        Wdr_I = alpha * normWCC @ Wdr_i @ normWDD + (1 - alpha) * Wdr0

    T_recovery = Wdr_I
    return T_recovery