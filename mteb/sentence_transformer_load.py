import numpy as np
import matplotlib.pyplot as plt

# GLOBAL VARIABLE
model_name = "all-MiniLM-L6-v2"
dataset_name_1 = "Banking77Classification"
dataset_name_2 = "STSBenchmark"

embedded_1 = np.load(f"embedded/{dataset_name_1}_{model_name}.npy")
embedded_2 = np.load(f"embedded/{dataset_name_2}_{model_name}.npy")
# embedded_concat = [x for pair in zip(embedded_1, embedded_2) for x in pair]

embedded_1 -= np.mean(embedded_1, axis=0, keepdims=True)
embedded_2 -= np.mean(embedded_2, axis=0, keepdims=True)
# embedded_concat -= np.mean(embedded_concat, axis=0, keepdims=True)

embedded_concat = np.vstack([embedded_1, embedded_2])


U1, S1, VT1 = np.linalg.svd(embedded_1, full_matrices=False)
V1 = VT1.T
U2, S2, VT2 = np.linalg.svd(embedded_2, full_matrices=False)
V2 = VT2.T

Uconcat, Sconcat, VTconcat = np.linalg.svd(embedded_concat, full_matrices=False)
Vconcat = VTconcat.T

g_x = [V1[0] @ embedded_1[i].T for i in range(len(embedded_1))]
g_y = [V1[1] @ embedded_1[i].T for i in range(len(embedded_1))]

plt.figure()
plt.title("embedded1: basis V[0] V[1]")
plt.scatter(g_x, g_y, alpha=0.5)
plt.savefig("outputs/svd_bank77.png")

g_x = [V2[0] @ embedded_2[i].T for i in range(len(embedded_2))]
g_y = [V2[1] @ embedded_2[i].T for i in range(len(embedded_2))]

plt.figure()
plt.title("embedded2: basis V[0] V[1]")
plt.scatter(g_x, g_y, alpha=0.5)
plt.savefig("outputs/svd_stsbench.png")

# g_x = [Vconcat[0] @ embedded_concat[i].T for i in range(len(embedded_concat))]
# g_y = [Vconcat[1] @ embedded_concat[i].T for i in range(len(embedded_concat))]

# plt.figure()
# plt.title("embedded_concat: basis V[0] V[1]")
# plt.scatter(g_x, g_y, alpha=0.5)
# plt.savefig("outputs/svd_stsbench.png")

proj_1 = embedded_1 @ Vconcat[:, :2]
proj_2 = embedded_2 @ Vconcat[:, :2]

plt.figure()
plt.title("embedded_concat: basis V[0] V[1]")
plt.scatter(proj_1[:, 0], proj_1[:, 1], alpha=0.5, label=dataset_name_1, c="blue")
plt.scatter(proj_2[:, 0], proj_2[:, 1], alpha=0.5, label=dataset_name_2, c="red")
plt.legend()
plt.savefig("outputs/svd_concat.png")