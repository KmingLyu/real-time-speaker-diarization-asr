import numpy as np
import torch
from typing import Optional, List, Iterable, Tuple
# from scipy.spatial.distance import cdist
from pyannote.core.utils.distance import cdist # 比 scipy 的 cdist 多了幾種 metric 可以用

class IncrementalSpeakerClustering:
    def __init__(
        self,
        delta_new: float,
        metric: Optional[str] = "cosine",
        max_speakers: int = 20
    ):
        self.delta_new = delta_new # 設定一個閾值來判斷是否需要新的 centroid
        self.metric = metric # 距離度量方式，預設為 "cosine"
        self.max_speakers = max_speakers # 最大的 global speakers 數量
        self.centers: Optional[np.ndarray] = None # 儲存所有 centroids 的 numpy 陣列
        self.counts = np.zeros(self.max_speakers, dtype=int) # 儲存每個 centroid 的計數器

    def get_next_center_position(self) -> Optional[int]:
        """尋找下一個可用的位置來儲存新的中心點。

        返回值
        ------
        next_position: int 或 None
            下一個可用位置的索引；如果沒有可用位置，則為 None。
        """
        # 找到第一個所有元素都為0的向量（即尚未被使用的位置）
        for i in range(self.max_speakers):
            if np.all(self.centers[i] == 0):
                return i
        return None
    
    def init_centers(self, dimension: int):
        """初始化 centroid 矩陣"""
        self.centers = np.zeros((self.max_speakers, dimension))

    def update(self, global_speaker: int, embedding: np.ndarray):
        """更新已知的 centroid"""
        assert global_speaker < len(self.centers), "Cannot update unknown centers"
        self.centers[global_speaker] += embedding

    def add_center(self, embedding: np.ndarray) -> int:
        """新增一個 centroid，並返回它的 index"""
        next_position = self.get_next_center_position()
    
        # 檢查是否還有可用位置
        if next_position is None or next_position >= self.max_speakers:
            raise IndexError("No available positions to add a new centroid.")
        self.centers[next_position] = embedding
        return next_position

    def identify_single_speaker(self, embedding: torch.Tensor) -> int:
        """識別單一 speaker 的 identity

        參數：
        embedding: torch.Tensor
            該 speaker 的 embedding

        返回：
        global_speaker: int
            該 speaker 對應的 global speaker index
        """

        # 如果還沒有 centroids，先初始化
        if self.centers is None:
            self.init_centers(embedding.shape[0])
            global_speaker = self.add_center(embedding)
            return global_speaker
        
        # 計算這個embedding與所有現有的centroids之間的距離
        distances = cdist(embedding.reshape(1, -1), self.centers, metric=self.metric).flatten()
        # print(f'\n\n\n{distances}\n\n\n\n')

        # 如果現有的 centroids 數量已經達到最大限制，我們只能將資料分配給最近的質心
        if self.get_next_center_position() is None:
            # 直接找到最近的質心並回傳
            global_speaker = np.argmin(distances)
            # 判斷是否需要更新該質心
            if distances[global_speaker] < self.delta_new:
                self.update(global_speaker, embedding)
            # 不更新該質心
            return global_speaker
        
        # 確認是否有現有的centroid與這個embedding的距離小於閾值
        valid_map = np.where(distances < self.delta_new)[0]
        # print(f'\n\n\n{valid_map}\n\n\n\n')

        # 若有距離小於閾值的centroid，則在這之中選擇最接近的一個
        if valid_map.size > 0:
            # 選擇最接近的一個
            global_speaker = valid_map[np.argmin(distances[valid_map])]
            # 更新該centroid
            self.update(global_speaker, embedding)
            return global_speaker
        
        global_speaker = self.add_center(embedding)
        self.counts[global_speaker] += 1

        return global_speaker

