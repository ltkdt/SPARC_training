import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np
from sklearn.manifold import TSNE

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=3, random_state=42)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure()

for class_value in np.unique(y):
    plt.scatter(
        X_tsne[y == class_value, 0],
        X_tsne[y == class_value, 1],
        label=f"Class {class_value}"
    )

plt.title("t-SNE visualization")
plt.legend()
plt.savefig('data_visualization.png')



