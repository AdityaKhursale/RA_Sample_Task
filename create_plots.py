import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def createDataset(xstart, xend, ystart, yend):
    x = []
    y = []
    p = [[0.00 for _ in range(0, 50)] for _ in range(0, 50)]


    for i in range(0, 200):
        x.append(random.randint(xstart, xend))

    for j in range(0, 200):
        y.append(random.randint(ystart, yend))

    for i in range(ystart, yend):
        for j in range(xstart, xend):
            p[i][j] = 0.025


    x = np.array(x)
    y = np.array(y)
    p = np.array(p)
    
    return x, y, p


def createSubPlot(sx, sy, x, y, p, title, majorTicks):
    sp = axs[sx, sy].pcolormesh(p, alpha=0.2, cmap='viridis', edgecolors="white")
    axs[sx, sy].set_xticks(majorTicks)
    axs[sx, sy].set_yticks(majorTicks)
    axs[sx, sy].minorticks_off()
    axs[sx, sy].scatter(x, y, c='black', s=1)
    axs[sx, sy].set_xlim(0, 50)
    axs[sx, sy].set_ylim(0, 50)
    axs[sx, sy].set_title(title)
    return sp


x2 = []
y2 = []
z2 = []


# Plot part A

majorTicks = np.arange(0, 50, 10)
fig, axs = plt.subplots(2, 2)

x, y, p = createDataset(0, 25, 25, 50)
x[198] = 13
x[199] = 46
y[198] = 13
y[199] = 18
sp = createSubPlot(0, 0, x, y, p, 't=[0, 3000]', majorTicks)
x2.extend(x)
y2.extend(y)

x, y, p = createDataset(25, 50, 25, 50)
x[198] = 13
x[199] = 46
y[198] = 46
y[199] = 5
sp = createSubPlot(0, 1, x, y, p, 't=[3001, 6000]', majorTicks)
x2.extend(x)
y2.extend(y)

x, y, p = createDataset(0, 25, 0, 25)
x[198] = 13
x[199] = 32
y[198] = 46
y[199] = 15
sp = createSubPlot(1, 0, x, y, p, 't=[6001, 9000]', majorTicks)
x2.extend(x)
y2.extend(y)

x, y, p = createDataset(25, 50, 0, 25)
x[198] = 13
x[199] = 32
y[198] = 46
y[199] = 32
sp = createSubPlot(1, 1, x, y, p, 't=[9001, 12000]', majorTicks)
x2.extend(x)
y2.extend(y)

cb = fig.colorbar(sp, label='P', ax=axs.ravel().tolist(), orientation="horizontal")
sp.set_clim(0.00, 0.02)
fig.set_figheight(10)
plt.savefig('part_a.png')
plt.clf()

# Plot part B

x2 = np.array(x2)
y2 = np.array(y2)
z2 = np.array([i for i in range(0, 800)])

fig = plt.figure()
ax = plt.axes(projection='3d')
fig.tight_layout()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

colors = ['red' for _ in range(0, 200)]
colors.extend(['green' for _ in range(200, 400)])
colors.extend(['blue' for _ in range(400, 600)])
colors.extend(['black' for _ in range(600, 800)])

x_scale=8
y_scale=3
z_scale=12

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj

fig.set_size_inches(5, 5)
ax.grid(True)
ax.scatter(x2, y2, z2, color=colors, s=1)
ax.invert_xaxis()
ax.invert_yaxis()
plt.savefig("part_b.png")
plt.clf()


# Plot part C
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)


X = np.vstack((x2, y2)).T
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)
plot_dendrogram(model, truncate_mode="level")
plt.axis('off')
plt.savefig("part_c.png")
plt.clf()
