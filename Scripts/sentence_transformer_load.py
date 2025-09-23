import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# GLOBAL VARIABLE
model_name = "all-MiniLM-L6-v2"
dataset_name_1 = "Banking77Classification"
dataset_name_2 = "STSBenchmark"

embedded_1 = np.load(f"embedded/{dataset_name_1}_{model_name}.npy")
embedded_2 = np.load(f"embedded/{dataset_name_2}_{model_name}.npy")

scaler = StandardScaler()
embedded_1 = scaler.fit_transform(embedded_1)
embedded_2 = scaler.fit_transform(embedded_2)

labels_true = np.array([0] * len(embedded_1) + [1] * len(embedded_2)) 

embedded_concat = np.vstack([embedded_1, embedded_2])

Uconcat, Sconcat, VTconcat = np.linalg.svd(embedded_concat, full_matrices=False)
Vconcat = VTconcat.T


# proj_1 = embedded_1 @ Vconcat[:, :2]
# proj_2 = embedded_2 @ Vconcat[:, :2]

# plt.figure()
# plt.title("embedded_concat: basis V[0] V[1]")
# plt.scatter(proj_1[:, 0], proj_1[:, 1], alpha=0.5, label=dataset_name_1, c="blue")
# plt.scatter(proj_2[:, 0], proj_2[:, 1], alpha=0.5, label=dataset_name_2, c="red")
# plt.legend()
# plt.savefig("outputs/svd_concat.png")


proj = TruncatedSVD(n_components=2).fit_transform(embedded_concat)
kmeans = KMeans(n_clusters=2).fit(embedded_concat)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Ground Truth Labels")
plt.scatter(proj[:, 0], proj[:, 1], c=labels_true, cmap='coolwarm', alpha=0.5)

plt.subplot(1, 2, 2)
plt.title("Expected Outcome")
plt.scatter(proj[:, 0], proj[:, 1], c=kmeans.labels_, cmap='coolwarm', alpha=0.5)

plt.tight_layout()
plt.savefig(f"outputs/compare_{dataset_name_1}_{dataset_name_2}_{model_name}.png")
