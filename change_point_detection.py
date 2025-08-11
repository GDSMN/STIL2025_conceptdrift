import numpy as np
import pandas as pd
import ruptures as rpt
from typing import List, Literal

class ChangePointDetector:
    """
    df : pandas.DataFrame
    date_col : str
    emb_col : str
    freq : str, opcional
        frequência de agregação padrão = 'D'
    model : str,
        modelo do ruptures = 'rbf'
    algo : {'binseg', 'pelt'}
        Algoritmo de detecção.
    pen : float | None,
        Penalidade (PELT ou Binseg).
    n_bkps : int | None,
        Nº de quebras a estimar se `pen` for None.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str,
        emb_col: str,
        freq: str = "D",
        model: str = "rbf",
        algo: Literal["binseg", "pelt"] = "binseg",
        pen: float | None = 3.0,
        n_bkps: int | None = None,
    ) -> None:
        self.df = df.copy()
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.date_col, self.emb_col = date_col, emb_col
        self.freq, self.model, self.algo = freq, model, algo
        self.pen, self.n_bkps = pen, n_bkps

    def _centroids(self) -> pd.Series:
        return (
            self.df.groupby(pd.Grouper(key=self.date_col, freq=self.freq))[self.emb_col]
            .apply(lambda embs: np.vstack(embs).mean(0))
            .dropna()
            .sort_index()
        )

    def detect(self) -> List[pd.Timestamp]:
        centroids = self._centroids()
        signal = np.vstack(centroids.values)

        if self.algo == "binseg":
            detector = rpt.Binseg(model=self.model).fit(signal)
        else:
            detector = rpt.Pelt(model=self.model).fit(signal)

        if self.pen is not None:
            idx = detector.predict(pen=self.pen)
        else:
            idx = detector.predict(n_bkps=self.n_bkps)

        return centroids.index[idx].tolist()