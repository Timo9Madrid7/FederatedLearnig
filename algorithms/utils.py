from typing import Union, Optional, List, Dict, Set
import numpy as np
from collections import deque
import heapq


class QuickSort:
    def __partition__(self, arr: Union[np.ndarray, List], l: int, r: int) -> int:
        i = l
        pivot = arr[r]

        for j in range(l, r):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        
        arr[i], arr[r] = arr[r], arr[i]
        return i
    

    def __quick_sort__(self, arr: List, l: int, r: int) -> None:
        if l < r:
            idx = self.__partition__(arr, l, r)
            self.__quick_sort__(arr, idx + 1, r)
            self.__quick_sort__(arr, l, idx - 1)


    def quickSort(self, arr: List) -> List:
        self.__quick_sort__(arr, 0, len(arr) - 1)
        return arr

    
    def kth_greatest(self, arr: List, k: int) -> int:
        l, r = 0, len(arr) - 1
        while l <= r:
            idx = self.__partition__(arr, l, r)
            if idx == len(arr) - k:
                return arr[idx]
            elif idx < len(arr) - k:
                l = idx + 1
            else:
                r = idx - 1
    

    def kth_smallest(self, arr: List, k: int) -> int:
        k -= 1
        l, r = 0, len(arr) - 1

        while l <= r:
            idx = self.__partition__(arr, l, r)
            if idx == k:
                return arr[idx]
            elif idx < k:
                l = idx + 1
            else:
                r = idx - 1


class MSTBuilder:
    def prims(self, graph: np.ndarray, start: int = 0) -> Dict[int, int]:
        visited = set()
        parent: Dict[int, int] = {}
        distance_to_tree: Dict[int, float] = {vertex: float('inf') for vertex in range(graph.shape[0])}
        distance_to_tree[start] = 0
        queue = [(distance_to_tree[start], start)]

        while queue:
            _, v = heapq.heappop(queue)
            if v in visited:
                continue

            visited.add(v)
            for u, dist in enumerate(graph[v]):
                if u not in visited and dist < distance_to_tree[u]:
                    distance_to_tree[u] = dist
                    parent[u] = v
                    heapq.heappush(queue, (dist, u))
            
            if len(visited) == graph.shape[0]:
                break

        return parent


class SingleLinkageTreeNode:
    def __init__(self, points = set()) -> None:
        self.parent: Optional[SingleLinkageTreeNode] = None
        self.left: Optional[SingleLinkageTreeNode] = None
        self.right: Optional[SingleLinkageTreeNode] = None
        self.points: Set[int] = points
        self.distance: float = None
        
        self.lambda_birth = 0
        self.lambda_p: Dict[int, float] = {}

class SLTBuilder():
    def __init__(self, edges: List[List[float]], n: int) -> None:
        leafmapper: Dict[int, SingleLinkageTreeNode] = {i: SingleLinkageTreeNode({i}) for i in range(n)}
        self._n_ = n
        self.__C__ = n - 1
        edges.sort()
        for dist, u, v in edges:
            self.root = self.__union__(leafmapper[u], leafmapper[v], dist)
        self.linkage = self.__getLinkage__()

    def __find__(self, u: SingleLinkageTreeNode) -> SingleLinkageTreeNode:
        if u.parent is None:
            return u
        return self.__find__(u.parent)

    def __union__(self, u: SingleLinkageTreeNode, v: SingleLinkageTreeNode, dist: float) -> SingleLinkageTreeNode:
        parent = SingleLinkageTreeNode()
        u_parent, v_parent = self.__find__(u), self.__find__(v)
        parent.left, parent.right, u_parent.parent, v_parent.parent = u_parent, v_parent, parent, parent
        parent.points = u_parent.points.union(v_parent.points)
        parent.distance = dist
        self.__C__ += 1
        return parent        

    def __getLinkage__(self) -> List[List[float]]:
        queue = deque([self.root])
        linkage = []
        while queue:
            node = queue.popleft()
            link = [None, None, None, None]
            if len(node.left.points) > 1:
                self.__C__ -= 1
                link[0] = self.__C__
                queue.append(node.left)
            else:
                link[0] = [point for point in node.left.points].pop()
            
            if len(node.right.points) > 1:
                self.__C__ -= 1
                link[1] = self.__C__
                queue.append(node.right)
            else:
                link[1] = [point for point in node.right.points].pop()

            link[2] = node.distance
            link[3] = len(node.points)

            linkage.append(link)

        linkage.reverse()
        return linkage



if __name__ == '__main__':
    from scipy.spatial import distance  
    import scipy.cluster.hierarchy as sch
    import matplotlib.pyplot as plt

    INF = float('inf')
    graph = np.array([
        [INF, 7, INF, 5, INF, INF, INF],
        [7, INF, 8, 9, 7, INF, INF],
        [INF, 8, INF, INF, 5, INF, INF],
        [5, 9, INF, INF, 15, 6, INF],
        [INF, 7, 5, 15, INF, 8, 9],
        [INF, INF, INF, 6, 8, INF, 11],
        [INF, INF, INF, INF, 9, 11, INF]
    ])
    mstBuilder = MSTBuilder()
    parents = mstBuilder.prims(graph=graph, start=3)
    edges = [[graph[vertex][parents[vertex]], vertex, parents[vertex]] for vertex in parents]
    sltBuilder = SLTBuilder(edges, graph.shape[0])
    # sch.dendrogram(sltBuilder.linkage)
    # plt.show()
    
    np.random.seed(2215)
    X = np.random.randint(0, 21, size=(8, 2))
    graph = distance.cdist(X, X, 'euclidean')
    mstBuilder = MSTBuilder()
    parents = mstBuilder.prims(graph=graph, start=0)
    edges = [[graph[vertex][parents[vertex]], vertex, parents[vertex]] for vertex in parents]
    sltBuilder = SLTBuilder(edges, graph.shape[0])
    print(sltBuilder.linkage)
    sch.dendrogram(sltBuilder.linkage)
    plt.show()