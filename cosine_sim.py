import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BiweeklyCosineDissimilarity:
    """
    df : pandas.DataFrame
    date_col : str
    emb_col : str (nome da coluna contendo os embeddings)
    freq : str, padrão '2W' (duas semanas).
    """

    def __init__(self, df: pd.DataFrame, date_col: str,
                 emb_col: str, freq: str = "2W") -> None:
        self.df = df.copy()
        self.date_col = date_col
        self.emb_col = emb_col
        self.freq = freq

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        self._centroids = None
        self._dissim    = None

    def compute(self) -> pd.Series:

        if self._dissim is None:
            self._centroids = (
                self.df
                .groupby(pd.Grouper(key=self.date_col, freq=self.freq))[self.emb_col]
                .apply(lambda vs: np.vstack(vs).mean(axis=0))
                .sort_index()
            )

            dates = self._centroids.index
            sims = [
                1 - cosine_similarity(
                    self._centroids.iloc[i - 1].reshape(1, -1),
                    self._centroids.iloc[i].reshape(1, -1)
                )[0, 0]
                for i in range(1, len(dates))
            ]
            self._dissim = pd.Series(sims, index=dates[1:])

        return self._dissim