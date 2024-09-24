import numpy as np
import pandas as pd
import math
import numpy.matlib
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy


circRNA_disease_M = np.loadtxt(r"association.txt", dtype=int)



#计算circRNA拉普拉斯核核相似性
def Laplace_circRNA():
    row=533
    a = 2

    SM1=np.matlib.zeros((row,row))

    for i in range(0,row):
        for j in range(0,row):
            SM1[i,j]=math.exp(-(1/a)*np.linalg.norm((circRNA_disease_M[i,]-circRNA_disease_M[j,])))


    GSM = SM1
    return GSM



#计算疾病拉普拉斯核核相似性
def Laplace_disease():
    column = 89
    a = 2

    miRNA1=np.matlib.zeros((column,column))

    for i in range(0,column):
        for j in range(0,column):
            miRNA1[i,j]=math.exp(-(1/a)*np.linalg.norm((circRNA_disease_M[:,i]-circRNA_disease_M[:,j])))


    GmiRNA = miRNA1
    return GmiRNA



def main():
    LKS_circRNA = Laplace_circRNA()
    LKS_disease = Laplace_disease()

    np.savetxt(r'LKS_circRNA.txt', LKS_circRNA, delimiter='\t', fmt='%.9f')
    np.savetxt(r'LKS_disease.txt',  LKS_disease, delimiter='\t', fmt='%.9f')


if __name__ == "__main__":

        main()