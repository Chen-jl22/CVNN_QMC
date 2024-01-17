import numpy as np
import scipy.stats as scs
from options import European_call_option


def get_normal_samples(dim, mean=None, cov=None, m=16, method='MC'): # 多维时，cov 要求矩阵数组
    n = 2 ** m # 样本个数
    if (method == 'MC') | (method == 'QMC'): # 涉及不到被积函数
        if dim == 1:
            if np.array([mean]).all() == None:
                mean = 0
            else:
                pass
            if np.array([cov]).all() == None:
                cov = 1
            else:
                pass
            if method == 'MC':
                samples = np.random.normal(mean, np.sqrt(cov), size=n)  
            else:
                samples = scs.qmc.MultivariateNormalQMC(mean=mean, cov=cov).random(n).flatten()
            return samples
        
        else: # 维数超过1
            if np.array([cov]).all() == None: # 协方差默认
                if np.array([mean]).all() == None:  # 均值默认
                    if method == 'MC':
                        samples = np.random.normal(0, 1, size=(n, dim))
                    else:
                        samples = scs.qmc.MultivariateNormalQMC(mean=np.zeros(dim), cov=np.eye(dim)).random(n).reshape((n, dim))
                else: # 均值不默认
                    mean = np.array(mean)
                    if method == 'MC':
                        samples = mean + np.random.normal(0, 1, size=(n, dim))
                    else:
                        samples = scs.qmc.MultivariateNormalQMC(mean=mean, cov=np.eye(dim)).random(n).reshape((n, dim))
    
            else: # 协方差不默认
                cov = np.array(cov)
                if np.array([mean]).all() == None:
                    mean = np.zeros(dim)
                else:
                    mean = np.array(mean)
                if method == 'MC':
                    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
                else:
                    samples = scs.qmc.MultivariateNormalQMC(mean=mean, cov=cov).random(n).reshape((n, dim))
            
                
#                     A = np.linalg.cholesky(cov) # 标准 STD 法（Cholesky 分解得到生成矩阵 A）
#                     # 主成分 PCA 法（特征向量分解得到生成矩阵 A）
#                     eigVals, eigVects = np.linalg.eig(cov) # 提取特征值与特征向量，使得 cov=PDP^T（P为正交矩阵）
#                     eigValInd = np.argsort(-eigVals) # 特征值由大到小排序做记号
#                     re_eigVals = eigVals[eigValInd] # 特征值由大到小重新排序，D'
#                     re_eigVects = eigVects[:, eigValInd] # 特征向量重新排序，P'使得 cov=P'D'P'^T（P'正交）
#                     A = np.dot(re_eigVects, np.diag(np.sqrt(re_eigVals))) # 得到分解矩阵 A=P'D'^{1/2}，使得 cov=AA^T                
                
            return samples
    else:
        print('Method is invalid!')

# 2.2 多元正态分布下积分
def f1(x):
    return np.sum(x)

def get__GPCA_samples(integrand, dim, mean=None, cov=None, m=16, method='MC', GPCA=False): # 被积函数的有效维数高时（如 np.exp(mean(X))）用 GPCA，但时间会长
    n = 2 ** m # 样本个数
    # print("using gPCA")
    if (method =='MC') | (method =='QMC'):
        if dim == 1: # 一维情况
            samples = get_normal_samples(dim=dim, mean=mean, cov=cov, m=m, method=method)
            integral = []
            for x in samples:
                integral = np.append(integral, integrand(x))
            integral = integral.mean()
            return integral
        else: # 多维情况
            if GPCA == True:       
                if np.array([mean]).all() == None:
                    mean = np.zeros(dim)
                else:
                    mean = np.array(mean)
                if np.array([cov]).all() == None:
                    cov = np.eye(dim)
                    A = np.eye(dim)
                else:
                    cov = np.array(cov)
                    A = np.linalg.cholesky(cov) # 标准 STD 法（Cholesky 分解得到生成矩阵 A）
                test_stand_samples = get_normal_samples(dim=dim, m=10, method=method) # 生成测试样本：标准正态分布
                def grad_square(x):
                    grad = np.zeros(dim)
                    for i in range(0, dim):
                        x_delta = x.copy()
                        #print(x_delta)
                        x_delta[i] += 1e-5
                        #print(x_delta)
                        #print(np.shape(np.dot(A,x_delta)))
                        #print(integrand(mean + np.dot(A, x_delta)))
                        #print(integrand(mean + np.dot(A, x_delta)) - integrand(mean + np.dot(A, x)))
                        grad[i] = (integrand(mean + np.dot(A, x_delta)) - integrand(mean + np.dot(A, x))) / (1e-5)
                        # print('grad',grad[i])
                    grad = grad.reshape((1, dim))
                    matrix = np.dot(grad.T, grad)
                    return matrix
                infoMatrix = np.apply_along_axis(grad_square, 1, test_stand_samples).sum(axis=0) # 线性信息矩阵
                U, S, Vh = np.linalg.svd(infoMatrix) # 奇异值分解
                samples = mean + np.dot(np.dot(A, U), get_normal_samples(dim=dim, m=m, method=method).T).T
                # integral = np.apply_along_axis(integrand, 1, samples).mean()
                # print("using gPCA")
                return samples
            elif GPCA == False:
                # samples = get_normal_samples(dim=dim, mean=mean, cov=cov, m=m, method=method)
                # integral = np.apply_along_axis(integrand, 1, samples).mean()
                print('Not GPCA!')
            else:
                print("Parameter GPCA can only be True or False!")
    else:
        print('Method is invalid!')
    # 注意，这里针对 MC 或 QMC，我们均可以用重要性抽样来方差缩减，但需要解方程得到最优漂移，且其效果不是很大。
    # 注意，针对 QMC，我们可以先条件光滑化将被积函数光滑，再进行积分估计，这样效率会更快。
