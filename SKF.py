#************************双向均匀归一化预处理****************************
import numpy as np

K1 = 7
K2 = 7


# 从txt文件中读取数据
def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            row = [float(x) for x in line.split()]
            data.append(row)
    return np.array(data)

# 列归一化
def column_normalize(matrix):
    normalized_matrix = np.zeros(matrix.shape)
    for j in range(matrix.shape[1]):
        column_sum = np.sum(matrix[:, j])
        normalized_matrix[:, j] = matrix[:, j] / column_sum
    return normalized_matrix

#W is the matrix which needs to be normalized列归一化
# def column_normalize(w):
#     m = w.shape[0]
#     p = np.zeros([m,m])
#     for i in range(m):
#         for j in range(m):
#             if i == j:
#                 p[i][j] = 1/2
#             elif np.sum(w[i,:])-w[i,i]>0:
#                 p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
#     return p

# 计算邻居集合N(包含自身)
def calculate_neighbors(S, k):
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)

    return N



# 行归一化(邻居归一化）
def row_normalization(S, N):

    result = np.zeros(S.shape)
    for i in range(S.shape[0]):
        num = 0
        denominator=np.sum(S[i, N[i]]) # 分母计算的是邻居的和


       # denominator= np.sum(S[i])   #分母计算的是一整行
        for j in range(S.shape[1]):
            if j in N[i]:
                if denominator != 0:
                    num = num + 1

                    result[i, j] =  S[i, j] / denominator
                else:
                    result[i, j] = 0  # Set to 0 or handle differently based on your requirements
            else:
                result[i, j] = 0
    return result

# # 文件路径
# file_path = "LKGIP_miRNA.txt"
# # 从txt文件中读取数据
# input_matrix = read_data_from_txt(file_path)
# 邻居个数,后期需要调参！！！#circRNA邻居个数250，disease邻居个数40

GKGIP_SM = np.loadtxt('GKGIP_circRNA.txt')
GKGIP_miRNA = np.loadtxt('GKGIP_disease.txt')
LKGIP_SM = np.loadtxt('LKGIP_circRNA.txt')
LKGIP_miRNA = np.loadtxt('LKGIP_disease.txt')
lty = np.loadtxt('SM-miRNA关联矩阵.txt')
# 计算邻居集合N
N1 = calculate_neighbors(GKGIP_SM, K1)
N2 = calculate_neighbors(GKGIP_miRNA, K2)
N3 = calculate_neighbors(LKGIP_SM, K1)
N4 = calculate_neighbors(LKGIP_miRNA, K2)
# 执行列归一化
GKGIP_SM_col = column_normalize(GKGIP_SM)
GKGIP_miRNA_col = column_normalize(GKGIP_miRNA)
LKGIP_SM_col = column_normalize(LKGIP_SM)
LKGIP_miRNA_col = column_normalize(LKGIP_miRNA)
# 执行行归一化
GKGIP_SM_row = row_normalization(GKGIP_SM, N1)
GKGIP_miRNA_row = row_normalization(GKGIP_miRNA, N2)
LKGIP_SM_row = row_normalization(LKGIP_SM, N3)
LKGIP_miRNA_row = row_normalization(LKGIP_miRNA, N4)
#第二步
circRNA_P1=GKGIP_SM_col
circRNA_P2=LKGIP_SM_col
circRNA_S1=GKGIP_SM_row
circRNA_S2=LKGIP_SM_row
alpha_1 =0.5
disease_P1=GKGIP_miRNA_col
disease_P2=LKGIP_miRNA_col
disease_S1=GKGIP_miRNA_row
disease_S2=LKGIP_miRNA_row
circRNA_P2_t=circRNA_P2
circRNA_P1_t=circRNA_P1
for i in range(1000):
    circRNA_p1=alpha_1*(circRNA_S1@(circRNA_P2_t/2)@circRNA_S1.T)+(1-alpha_1)*(circRNA_P2/2)
    circRNA_p2=alpha_1*(circRNA_S2@(circRNA_P1_t/2)@circRNA_S2.T)+(1-alpha_1)*(circRNA_P1/2)
    err1 = np.sum(np.square(circRNA_p1-circRNA_P1_t))
    err2= np.sum(np.square(circRNA_p2-circRNA_P2_t))
    if (err1 < 1e-6) and (err2 < 1e-6):
        print("circRNA迭代的次数：",i)
        break
    circRNA_P2_t=circRNA_p2
    circRNA_P1_t=circRNA_p1
# #简单平均
circRNA_sl=0.5*circRNA_p1+0.5*circRNA_p2

# np.savetxt('SM_sl66666666666.txt', circRNA_sl,fmt='%6f',delimiter='\t')
# np.savetxt('circRNA_sl_p1.txt', circRNA_p1,fmt='%f',delimiter='\t')
# np.savetxt('circRNA_sl_p2.txt', circRNA_p2,fmt='%f',delimiter='\t')
#*******************************************************************************************
disease_P2_t=disease_P2
disease_P1_t=disease_P1
for j in range(1000):
    disease_p1=alpha_1*(disease_S1@(disease_P2_t/2)@disease_S1.T)+(1-alpha_1)*(disease_P2/2)
    disease_p2=alpha_1*(disease_S2@(disease_P1_t/2)@disease_S2.T)+(1-alpha_1)*(disease_P1/2)
    err1 = np.sum(np.square(disease_p1-disease_P1_t))
    err2= np.sum(np.square(disease_p2-disease_P2_t))
    if (err1 < 1e-6) and (err2 < 1e-6):
        print("disease迭代的次数：", i)
        break
    disease_P2_t=disease_p2
    disease_P1_t=disease_p1
disease_sl=0.5*disease_p1+0.5*disease_p2



def calculate_neighbors(S, k):
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)
    return N
def compute_weighted_matrix(S1, k1):
    # 计算邻居集合 N_j 和 N_i
    N_i = calculate_neighbors(S1, k1)  # 行的邻居集合
    N_j = calculate_neighbors(S1.T, k1)  # 列的邻居集合
    # 生成 w 矩阵
    w = np.zeros((len(S1), len(S1)))

    for i in range(len(S1)):
        for j in range(len(S1)):
            if i in N_j[j] and j in N_i[i]:
                w[i][j] = 1
            elif i not in N_j[j] and j not in N_i[i]:
                w[i][j] = 0
            else:
                w[i][j] = 0.5
    return w
# 示例使用
w1 = compute_weighted_matrix(circRNA_sl, 7)
w2 = compute_weighted_matrix(disease_sl, 7)

# np.savetxt('W1.txt', w1,fmt='%f',delimiter='\t')
# np.savetxt('W2.txt', w2,fmt='%f',delimiter='\t')
average_circRNA = w1 @ circRNA_sl
average_disease = w2 @ disease_sl
np.savetxt('circRNA_SMF.txt',average_circRNA,fmt='%6f',delimiter='\t')
np.savetxt('disease_SMF.txt',average_disease,fmt='%6f',delimiter='\t')
# np.savetxt('miRNA_sl666666666.txt', disease_sl,fmt='%6f',delimiter='\t')
# # 保存列归一化结果到txt文件
# column_normalized_file_path = "LKGIP_miRNA_col666.txt"
# np.savetxt(column_normalized_file_path, column_normalized_matrix, delimiter='\t',fmt='%.8f')
# # 保存行归一化结果到txt文件
# row_normalized_file_path = "LKGIP_miRNA_row666.txt"
# np.savetxt(row_normalized_file_path, row_normalized_matrix, delimiter='\t',fmt='%.8f')
# print("列归一化结果已保存到文件:", column_normalized_file_path)
# print("行归一化结果已保存到文件:", row_normalized_file_path)
# #************************************************************************

