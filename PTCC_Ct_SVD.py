dim1 = 10000  # 路段ID
dim2 = 20000  # 用户ID
dim3 = 8  # 时间段
history_dim = 4 # 历史

import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
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

# 定义Traffic conditions
T_dr = A[:, :, : history_dim]
T_dh = A[:, :, history_dim:]

import numpy as np
from sklearn.cluster import KMeans



import torch
from tqdm import tqdm

def traffic_conditions_cluster(T_dr, I, M, K, el_th):
    # 初始化输出张量
    T_ct_cx = torch.zeros((I, M, K))
    P_ct_cx = torch.zeros((I, M, K))

    # 对于每个时间段k
    for k in tqdm(range(K)):
        # 对于每个路段i
        for i in range(I):
            # 提取所有非零行驶时间值
            vec = T_dr[i, :, k][T_dr[i, :, k] != 0]
            if vec.numel() > el_th:
                # 如果vec非空，进行聚类
                kmeans = KMeans(n_clusters=M, random_state=0, n_init='auto').fit(vec.reshape(-1, 1))
                centroids = kmeans.cluster_centers_.flatten()  # centroids 聚类中心点  element还是时间
                labels = kmeans.labels_

                # 对每个类别计算样本数量和样本比例
                for m in range(M):
                    samples_m = np.sum(labels == m)
                    T_ct_cx[i, m, k] = centroids[m] if samples_m > 0 else 0
                    P_ct_cx[i, m, k] = samples_m / len(vec) if len(vec) > 0 else 0

    return T_ct_cx, P_ct_cx


# 假设I, M, K已经定义，代表路段数量、交通条件类别数量、时间段数量
I, _, K = dim1, dim2, int(dim3 / 2)
M = 15  # 聚类个数
el_th = 15  # el_th >= M
T_dr_ct_cx, P_dr_ct_cx = traffic_conditions_cluster(T_dr, I, M, K, el_th)
T_dh_ct_cx, P_dh_ct_cx = traffic_conditions_cluster(T_dh, I, M, K, el_th)

Tcl = torch.cat((T_dh_ct_cx, T_dr_ct_cx), dim=3)


def split_test(T):
    # 获取最后一个时间片对应的2D张量
    last_time_slice = T[:, :, -1]

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
    return test_set_indices, test_set_values


def svd_gradient_descent(R, alpha, beta, epsilon, max_epochs):
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

    epochs = 0

    while abs(loss_t - loss_t1) >= epsilon and epochs <= max_epochs:
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
        epochs += 1

    return P, Q


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

def calculate_rr(predicted):
    return np.count_nonzero(predicted)/predicted.size

def calculate_r_squared(actual, predicted):
    return 1 - np.sum((predicted - actual) ** 2) / np.sum((actual - np.mean(actual)) ** 2)

def calculate_mae(actual, predicted):
    return np.mean(np.abs(predicted - actual))

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((predicted - actual) ** 2))

def calculate_ec(actual, predicted):
    n = len(actual)
    numerator = np.sqrt(np.sum((actual - predicted) ** 2))
    denominator = np.sqrt(np.sum(actual ** 2) + (np.sum(actual) ** 2) / n)
    return 1 - (numerator / denominator)

def get_experimental_index(test_set_values, predicted_values):
    rr = calculate_rr(test_set_values, predicted_values)
    r_squared = calculate_r_squared(test_set_values, predicted_values)
    mae = calculate_mae(test_set_values, predict_set_values)
    mape = calculate_mape(test_set_values, predict_set_values)
    rmse = calculate_rmse(test_set_values, predict_set_values)
    ec = calculate_ec(test_set_values, predict_set_values)
    print(rr, r_squared, mae, mape, rmse, ec)



test_set_indices, test_set_values = split_test(T_dr_ct_cx)
test_set_values = test_set_values.numpy()



# 检查CUDA是否可用
cuda = torch.cuda.is_available()
print("CUDA Available:", cuda)

# 参数

alpha = 0.001
beta = 0.001
epsilon = 0.001
max_epochs = 1000

# 在GPU上并行执行SVD分解

for i in range(10):
    A_hat = torch.zeros(I, M, K)
    if cuda:
        A_hat = A_hat.cuda()

    for dim1_index in tqdm(range(I)):
        # 从A中获取一个二维切片
        slice_2d = T_dr_ct_cx[dim1_index, :, :]
        # slice_2d_fft = torch.fft.fft2(slice_2d)

        # 在GPU上执行SVD分解
        # P, Q = svd_gradient_descent_1(slice_2d, alpha, beta, max_epochs)
        P, Q = svd_gradient_descent(slice_2d, alpha, beta, epsilon, max_epochs)
        # 重构切片并更新到A_hat中
        A_hat[dim1_index, :, :] = torch.matmul(P, Q.T)

    # 将A_hat从GPU移动回CPU
    if cuda:
        A_hat = A_hat.cpu()

    predict_set_values = A_hat[test_set_indices[:, 0], test_set_indices[:, 1], -1]
    mean_val = torch.nanmean(predict_set_values)
    predict_set_values[torch.isnan(predict_set_values)] = 0
    predict_set_values[torch.isinf(predict_set_values)] = 0
    predict_set_values = predict_set_values.numpy()
    get_experimental_index(test_set_values, predict_set_values)


