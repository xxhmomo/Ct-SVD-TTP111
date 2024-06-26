import time

import tensorly
import numpy as np
import torch


def couple_t_svd_1(completed_A, context_tensor_X, context_tensor_Y, epsilon, rank_r, iterations_slice, order=2):
    """
        completed_A： 被补全张量
        context_tensor： 上下文张量
        epsilon: 每次couple_SVD分解的误差
        order； couple_T_SVD 分解的阶 默认等于1  order = dim2
        iterations_slice： 每个slice的最大迭代次数
    """
    # dim1 路段ID， dim2 用户ID  dim3 时间段
    dim1, dim2, dim3 = completed_A.shape
    x_dim1, x_dim2, x_dim3 = context_tensor_X.shape  # x_dim1 属性列   x_dim2  用户ID， x_dim3 时间段

    indices = torch.nonzero(completed_A)
    # get non-zero values
    values = completed_A[indices[:, 0], indices[:, 1], indices[:, 2]]
    turn = list(range(len(values)))

    # 进行couple——T——SVD分解
    completed_A_fft = torch.fft.fft(completed_A)  # dim1 * dim2
    context_tensor_X_fft = torch.fft.fft(context_tensor_X)  # x_dim1 * x_dim2

    lambda1 = 0.01
    lambda2 = 0.01

    # step size
    t0 = 100000000000
    t = t0

    # initialize function loss
    loss_t = epsilon + 1
    loss_t1 = 0

    for index in range(completed_A[order]):  # 按dim3进行耦合
        """
            completed_A_fft_slice:  dim1 * dim2
            context_tensor_X_fft_slice: x_dim1 * x_dim2 
            其中  dim2 == x_dim2
        """
        completed_A_fft_slice = completed_A_fft[:, :, index]

        context_tensor_X_fft_slice = context_tensor_X_fft[:, :, index]

        indices = torch.nonzero(completed_A_fft_slice)
        # get non-zero values
        values = completed_A_fft_slice[indices[:, 0], indices[:, 1]]
        turn = list(range(len(values)))

        # 初始化R矩阵
        U = torch.randn(dim1, rank_r)
        V = torch.randn(rank_r, dim2)
        R = torch.randn(x_dim1, rank_r)
        # 进行couple_SVD分解
        iterations_slice_temp = iterations_slice
        while iterations_slice and abs(loss_t - loss_t1) >= epsilon:

            nita = 1 / np.sqrt(t)
            t = t + 1

            for num in range(len(values) - 1):
                change = np.random.randint(num + 1, len(values))
                temp = turn[num]
                turn[num] = turn[change]
                turn[change] = temp

            for num in range(len(values)):
                tnum = turn[num]
                i = indices[tnum, 0]
                j = indices[tnum, 1]

                Ui = U[i, :]
                Vj = V[:, j]

                # U的梯度
                dFU = ((Ui @ Vj) - values[num]) @ Vj + lambda2 * Ui
                dFV = ((Ui @ Vj) - values[num]) @ Ui + lambda1 * (R @ Vj - values[num]) @ R + lambda2 * Vj
                dFR = (R @ Vj - values[num]) @ Vj + lambda2 * R

                U[i, :] = U - nita * dFU
                V[:, j] = V - nita * dFV
                R = R - nita * dFR

            c = torch.zeros(len(values))
            for j in range(len(values)):
                ij = U[indices[j, 0], 0] * V[0, indices[j, 1]]
                c[j] = ij
            loss_t = loss_t1
            loss_t1 = torch.norm(c - values)
            iterations_slice_temp -= 1


def svd_gradient_descent(R, alpha, beta, epsilon):
    """
        SVD Learning Using Gradient Descent with Regularization in PyTorch for GPU acceleration.

        :param R: Rating Matrix (PyTorch tensor)
        :param alpha: Learning Rate
        :param beta: Regularization parameter
        :param max_epochs: Maximum number of epochs
        :return: Factorized Matrices P and Q
        """
    # 先进行SVD分解 Initialize P and Q randomly on GPU
    num_users, num_items = R.shape
    num_latent_features = 10  # Can be adjusted

    # Initialize P and Q randomly on GPU
    P = torch.rand(num_users, num_latent_features, device=R.device)
    Q = torch.rand(num_items, num_latent_features, device=R.device)

    indices = torch.nonzero(R)
    values = R[indices[:, 0], indices[:, 1]]
    turn = list(range(len(values)))

    loss_t = epsilon + 1
    loss_t1 = 0

    while abs(loss_t - loss_t1) >= epsilon:
        # 打乱
        for num in range(len(values) - 1):
            change = np.random.randint(num + 1, len(values))
            temp = turn[num]
            turn[num] = turn[change]
            turn[change] = temp

        for num in range(len(values)):
            tnum = turn[num]

            i = indices[tnum, 0]
            j = indices[tnum, 1]
            Pi = P[i, :]
            Qj = Q[j, :]
            Pi = Pi.unsqueeze(0)
            Qj = Qj.unsqueeze(1)

            eij = Pi @ Qj - R[i, j]

            DFP = eij * (Qj.reshape(1, -1)) + alpha * Pi
            DFQ = eij * (Pi.reshape(-1, 1)) + alpha * Qj

            P[i, :] = (Pi - beta * DFP).squeeze()
            Q[j, :] = (Qj - beta * DFQ).squeeze()

        c = torch.zeros(len(values), device=R.device)
        R_rec = P @ torch.transpose(Q, 0, 1)
        for j in range(len(values)):
            c[j] = R_rec[indices[j, 0], indices[j, 1]]
        loss_t = loss_t1
        loss_t1 = torch.norm(values - c)

    return P, Q


def split_train_test(R, test_size=0.2):
    """
    Splits the rating matrix R into training and testing sets.
    """
    train = R.copy()
    test = np.zeros(R.shape)

    for user in range(R.shape[0]):
        test_ratings = np.random.choice(R[user, :].nonzero()[0],
                                        size=int(test_size * R[user, :].nonzero()[0].size),
                                        replace=False)
        train[user, test_ratings] = 0
        test[user, test_ratings] = R[user, test_ratings]

    # Ensure training and testing sets are disjoint
    assert (np.all((train * test) == 0))

    return train, test


dim1 = 4000  # 路段ID
dim2 = 5000  # 用户ID
dim3 = 8  # 时间段

import numpy as np
import torch

# Load data
tensor_Ar_Ah_init = np.loadtxt('../../Data/TensorData/tensor.txt', dtype=np.float_, delimiter=",")

# Subtract 1 时隙id下标从0开始
tensor_Ar_Ah_init[:, 1] = tensor_Ar_Ah_init[:, 1] - 1

# 交换第二列和第三列
tensor_Ar_Ah_init[:, [1, 2]] = tensor_Ar_Ah_init[:, [2, 1]]

# 路段ID、用户ID, 时间段、旅行时间
# 提取索引‘
indices = tensor_Ar_Ah_init[:, 0: 3].astype(int)
values = tensor_Ar_Ah_init[:, 3]

indices_indices = np.where((indices[:, 0] < dim1) & (indices[:, 1] < dim2) & (indices[:, 2] < dim3))

indices = indices[indices_indices]
values = values[indices_indices]

# 创建张量
A = torch.zeros((dim1, dim2, dim3))
for i in range(len(indices)):
    A[indices[i, 0], indices[i, 1], indices[i, 2]] = values[i]

# 获取最后一个时间片对应的2D张量
last_time_slice = A[:, :, -1]

# 获取2D张量中非零元素的索引
non_zero_indices = torch.nonzero(last_time_slice)
print("最后一个slice上面的非零元素数量：{}".format(len(non_zero_indices)))

# 从非零元素中随机选择30%作为测试集
num_non_zero = non_zero_indices.size(0)
num_test = int(0.3 * num_non_zero)
test_indices = torch.randperm(num_non_zero)[:num_test]
test_set_indices = non_zero_indices[test_indices]

# 生成测试集
test_set_values = last_time_slice[test_set_indices[:, 0], test_set_indices[:, 1]]

# 将测试集的索引处 A 的元素设置为 0
for idx in test_set_indices:
    batch_idx, row_idx = idx
    A[batch_idx, row_idx, -1] = 0
print("最后一个slice上面的非零元素数量：{}".format(len(torch.nonzero(A[:, :, -1]))))

import torch

# 检查CUDA是否可用
cuda = torch.cuda.is_available()
print("CUDA Available:", cuda)

# 参数

alpha = 0.001
beta = 0.001
max_epochs = 20
epsilon = 0.01

# 在GPU上并行执行SVD分解
A_hat = torch.zeros(dim1, dim2, dim3)
if cuda:
    A_hat = A_hat.cuda()
    A = A.cuda()

for dim1_index in range(dim1):
    # 从A中获取一个二维切片
    slice_2d = A[dim1_index, :, :]

    # 在GPU上执行SVD分解
    # P, Q = svd_gradient_descent_1(slice_2d, alpha, beta, max_epochs)
    P, Q = svd_gradient_descent(slice_2d, alpha, beta, epsilon)

    # 重构切片并更新到A_hat中
    A_hat[dim1_index, :, :] = torch.matmul(P, Q.T)
    print(dim1_index)

# 将A_hat从GPU移动回CPU
if cuda:
    A_hat = A_hat.cpu()

# A_hat现在包含重构的切片
