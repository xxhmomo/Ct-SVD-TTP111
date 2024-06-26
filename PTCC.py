import numpy as np
from sklearn.cluster import KMeans

# 假设T_drv是已经加载到内存中的张量
# T_drv = ...

# 假设I, M, K已经定义，代表路段数量、交通条件类别数量、时间段数量
# I, M, K = ...

# 假设th是行驶时间的阈值
# th = ...

# 初始化输出张量
T_ct_cx = np.zeros((I, M, K))
P_ct_cx = np.zeros((I, M, K))

# 对于每个时间段k
for k in range(K):
    # 对于每个路段i
    for i in range(I):
        # 提取所有非零行驶时间值
        vec = T_drv[i, :, k][T_drv[i, :, k] != 0]
        if vec.size > 0:
            # 如果vec非空，进行聚类
            kmeans = KMeans(n_clusters=M, random_state=0).fit(vec.reshape(-1, 1))
            centroids = kmeans.cluster_centers_.flatten()
            labels = kmeans.labels_

            # 对每个类别计算样本数量和样本比例
            for m in range(M):
                samples_m = np.sum(labels == m)
                T_ct_cx[i, m, k] = centroids[m] if samples_m > 0 else 0
                P_ct_cx[i, m, k] = samples_m / len(vec) if len(vec) > 0 else 0

# 返回结果
# T_ct_cx 和 P_ct_cx 就是最后的结果
