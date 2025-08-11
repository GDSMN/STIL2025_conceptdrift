import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BiweeklyCumulativeDissimilarity:
    """
    df : pandas.DataFrame
    date_col : str
    emb_col : str
    freq : str, opcional
    """

    def __init__(self, df: pd.DataFrame, date_col: str,
                 emb_col: str, freq: str = "2W") -> None:
        self.df = df.copy()
        self.date_col = date_col
        self.emb_col = emb_col
        self.freq = freq

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        self._centroids = None
        self._cum_dissim = None

    def compute(self) -> pd.Series:
        if self._cum_dissim is None:
            self._centroids = (
                self.df
                .groupby(pd.Grouper(key=self.date_col, freq=self.freq))[self.emb_col]
                .apply(lambda v: np.vstack(v).mean(axis=0))
                .sort_index()
            )

            dates = self._centroids.index
            n = len(dates)
            cum_dist = np.full(n, np.nan)

            baseline_sum = self._centroids.iloc[0].copy()

            for i in range(1, n):
                baseline_mean = baseline_sum / i
                current = self._centroids.iloc[i]

                sim = cosine_similarity(
                    baseline_mean.reshape(1, -1),
                    current.reshape(1, -1)
                )[0, 0]
                cum_dist[i] = 1 - sim
                baseline_sum += current

            self._cum_dissim = pd.Series(cum_dist, index=dates)

        return self._cum_dissim