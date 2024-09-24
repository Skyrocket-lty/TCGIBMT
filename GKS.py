import numpy as np
import pandas as pd
import math
import numpy.matlib
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy


circRNA_disease_M = np.loadtxt(r"association.txt", dtype=int)




#计算circRNA高斯轮廓核相似性

def Gaussian_circRNA():
    row=533
    sum=0
    CC1=np.matlib.zeros((row,row))
    for i in range(0,row):
        a=np.linalg.norm(circRNA_disease_M[i,])*np.linalg.norm(circRNA_disease_M[i,])
        sum=sum+a
    ps=row/sum
    for i in range(0,row):
        for j in range(0,row):
            CC1[i,j]=math.exp(-ps*np.linalg.norm(circRNA_disease_M[i,]-circRNA_disease_M[j,])*np.linalg.norm(circRNA_disease_M[i,]-circRNA_disease_M[j,]))

    CC = CC1
    return CC


#计算疾病高斯轮廓核相似性
def Gaussian_disease():
    column=89
    sum=0
    DD1=np.matlib.zeros((column,column))
    for i in range(0,column):
        a=np.linalg.norm(circRNA_disease_M[:,i])*np.linalg.norm(circRNA_disease_M[:,i])
        sum=sum+a
    ps=column/sum
    for i in range(0,column):
        for j in range(0,column):
            DD1[i,j]=math.exp(-ps*np.linalg.norm(circRNA_disease_M[:,i]-circRNA_disease_M[:,j])*np.linalg.norm(circRNA_disease_M[:,i]-circRNA_disease_M[:,j]))


    DD = DD1
    return DD


def main():
    GKS_circRNA = Gaussian_circRNA()
    GKS_disease = Gaussian_disease()

    np.savetxt(r'GKS_circRNA.txt', GKS_circRNA, delimiter='\t', fmt='%.9f')
    np.savetxt(r'GKS_disease.txt',  GKS_disease, delimiter='\t', fmt='%.9f')


if __name__ == "__main__":

        main()