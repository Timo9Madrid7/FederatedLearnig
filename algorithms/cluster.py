from typing import List, Dict, Set, Tuple, Callable, Any
import numpy as np
try:
    from .utils import QuickSort, MSTBuilder, SLTBuilder, SingleLinkageTreeNode, CTBuilder, CondenseTreeNode
except ImportError:
    from utils import QuickSort, MSTBuilder, SLTBuilder, SingleLinkageTreeNode, CTBuilder, CondenseTreeNode
import heapq
from scipy.cluster.hierarchy import dendrogram
from hdbscan.plots import SingleLinkageTree as Dendrogram
import matplotlib.pyplot as plt


class DBSCAN:
    def __init__(self, eps: float, min_samples: int, distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.square(x - y).sum()) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self._distance_metric_ = distance_metric


    def __find_neighbors__(self, data: np.ndarray) -> None:
        self._neighbors_: Dict[int, set] = {idx: {idx} for idx in range(self._n_)}
        for idx in range(self._n_):
            for jdx in range(idx + 1, self._n_):
                if self._distance_metric_(data[idx], data[jdx]) <= self.eps * self.eps:
                    self._neighbors_[idx].add(jdx)
                    self._neighbors_[jdx].add(idx)


    def __form_clusters__(self) -> None:
        C = 0
        for idx in range(self._n_):
            if self.labels_[idx] >= 0:
                continue
            if len(self._neighbors_[idx]) >= self.min_samples:
                self.__expand_cluster__(idx, C)
                C += 1


    def __expand_cluster__(self, idx: int, C: int) -> None:
        self.labels_[idx] = C
        for jdx in self._neighbors_[idx]:
            if self.labels_[jdx] >= 0:
                continue
            self.labels_[jdx] = C
            if len(self._neighbors_[jdx]) >= self.min_samples:
                self.__expand_cluster__(jdx, C)


    def fit(self, data: np.ndarray) -> None:
        self._n_ = data.shape[0]
        self.labels_ = -1 * np.ones(self._n_, dtype=np.int32)
        self.__find_neighbors__(data)
        self.__form_clusters__()


    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.labels_


class HDBSCAN:
    def __init__(self, min_samples: int, min_cluster_size: int, distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y)) -> None:
        self.min_samples = min_samples
        self.min_cluster_size = max(2, min_cluster_size)
        self._distance_metric_ = distance_metric


    def __transform_space__(self, data: np.ndarray) -> None:
        distance_matrix: List[List[float]] = [[float('inf') for _ in range(self._n_)] for _ in range(self._n_)]
        for idx in range(self._n_):
            for jdx in range(idx + 1, self._n_):
                distance_matrix[idx][jdx] = distance_matrix[jdx][idx] = self._distance_metric_(data[idx], data[jdx])

        quicksort = QuickSort()
        core_distance: List[float] = [None for _ in range(self._n_)]
        for idx in range(self._n_):
            core_distance[idx] = quicksort.kth_smallest(distance_matrix[idx].copy(), self.min_samples) # TODO: KD-Tree

        self._mutual_reachability_distance_: List[List[float]] = [[float('inf') for _ in range(self._n_)] for _ in range(self._n_)]
        for idx in range(self._n_):
            for jdx in range(idx + 1, self._n_):
                self._mutual_reachability_distance_[idx][jdx] \
                    = self._mutual_reachability_distance_[jdx][idx] \
                        = max(core_distance[idx], core_distance[jdx], distance_matrix[idx][jdx])


    def __build_minimum_spanning_tree__(self) -> Dict[int, int]:
        mstBuilder = MSTBuilder()
        parents = mstBuilder.prims(np.array(self._mutual_reachability_distance_))
        return parents
    

    def __build_cluster_hierarchy__(self, parents: Dict[int, int]) -> SLTBuilder:
        edges = [[self._mutual_reachability_distance_[vertex][parents[vertex]], vertex, parents[vertex]] for vertex in parents]
        sltBuilder = SLTBuilder(edges, self._n_)
        return sltBuilder
    

    def __condense_cluster_tree__(self, sltBuilder: SLTBuilder) -> CTBuilder:
        ctBuilder = CTBuilder()
        ctBuilder.condenseSLT(sltBuilder.root, self.min_cluster_size)
        return ctBuilder
        

    def __extract_clusters__(self, ctBuilder: CTBuilder) -> None:
        pq = ctBuilder.leaves.copy()
        sel = ctBuilder.selected
        visited: Set[CondenseTreeNode] = set()

        def unselect_descendants(node: CondenseTreeNode) -> None:
            if node.left:
                sel.discard(node.left)
                unselect_descendants(node.left)
            if node.right:
                sel.discard(node.right)
                unselect_descendants(node.right)

        while pq:
            node = heapq.heappop(pq)
            if node in visited:
                continue
            sibling = node.parent.right if node is node.parent.left else node.parent.left
            visited.add(node)
            visited.add(sibling)

            if sibling.stability + node.stability > node.parent.stability:
                node.parent.set_stability(sibling.stability + node.stability)
            else:
                unselect_descendants(node.parent)
                sel.add(node.parent)

            if not node.parent is ctBuilder.root:
                heapq.heappush(pq, node.parent)


    def fit(self, data: np.ndarray) -> None:
        self._n_ = data.shape[0]
        self.__transform_space__(data)
        parents = self.__build_minimum_spanning_tree__()
        sltBuilder = self.__build_cluster_hierarchy__(parents)
        self._linkage_ = sltBuilder.linkage
        ctBuilder = self.__condense_cluster_tree__(sltBuilder)
        self.__extract_clusters__(ctBuilder)
        self.labels_ = -1 * np.ones(data.shape[0], dtype=int)
        for c_id, c in enumerate(ctBuilder.selected):
            self.labels_[list(c.lambda_p.keys())] = c_id

    
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.labels_


    def dendrogram(self) -> None:
        # color_dict = dendrogram(self._linkage_)
        plot = Dendrogram(self._linkage_)
        plot.plot()
        plt.show()



if __name__ == '__main__':
    from sklearn import cluster, datasets
    from tqdm import tqdm
    import random
    if False:
        for _ in tqdm(range(100)):
            x, y = datasets.make_blobs(n_samples=random.randint(50, 200), n_features=random.randint(2, 10))
            dbscan = cluster.DBSCAN()
            mydbscan = DBSCAN(0.5, 5)

            dbscan.fit(x)
            mydbscan.fit(x)

            assert (dbscan.labels_ == mydbscan.labels_).all()
        print("pass all test cases")

    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from hdbscan import HDBSCAN as HDBSCAN_ORI
    import numpy as np
    X, y = make_blobs(50, centers=[[-2,-2], [1,1], [3,0]], cluster_std=0.4, random_state=40)

    hdbscan = HDBSCAN_ORI(min_samples=2, min_cluster_size=5, cluster_selection_epsilon=0)
    hdbscan.fit(X)
    plt.scatter(X[:,0], X[:,1], c=hdbscan.labels_, cmap="rainbow")
    plt.show()

    myhdbscan = HDBSCAN(min_samples=2, min_cluster_size=5)
    myhdbscan.fit(X)
    plt.scatter(X[:,0], X[:,1], c=myhdbscan.labels_, cmap="rainbow")
    plt.show()
