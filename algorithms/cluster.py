from typing import Dict, Callable
import numpy as np

class DBSCAN:
    def __init__(self, eps: float, min_samples: int, distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.square(x - y).sum()) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self._distance_metric_ = distance_metric

    def __find_neighbors__(self) -> None:
        self._neighbors_: Dict[int, set] = {idx: {idx} for idx in range(self._n_)}
        for idx in range(self._n_):
            for jdx in range(idx + 1, self._n_):
                if self._distance_metric_(self._data_[idx], self._data_[jdx]) <= self.eps * self.eps:
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
        self._data_ = data
        self._n_ = self._data_.shape[0]
        self.labels_ = -1 * np.ones(self._n_, dtype=np.int32)
        self.__find_neighbors__()
        self.__form_clusters__()

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.labels_


if __name__ == '__main__':
    from sklearn import cluster, datasets
    from tqdm import tqdm
    import random
    for _ in tqdm(range(100)):
        x, y = datasets.make_blobs(n_samples=random.randint(50, 200), n_features=random.randint(2, 10))
        dbscan = cluster.DBSCAN()
        mydbscan = DBSCAN(0.5, 5)

        dbscan.fit(x)
        mydbscan.fit(x)

        assert (dbscan.labels_ == mydbscan.labels_).all()
    print("pass all test cases")



    