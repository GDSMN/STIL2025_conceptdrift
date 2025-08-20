import pandas as pd
import numpy as np
from alibi_detect.cd import MMDDrift

class BiweeklyMMD:
    """
    df : pandas.DataFrame
    date_col : str
    emb_col : str
    freq : str, opcional
    backend : {'pytorch', 'tensorflow', 'keops'},
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str,
        emb_col: str,
        freq: str = "2W",
        backend: str= "pytorch",
        p_val: float = 0.05,
    ) -> None:
        self.df = df.copy()
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.date_col, self.emb_col = date_col, emb_col
        self.freq, self.backend, self.p_val = freq, backend, p_val

    def compute(self) -> pd.Series:
        grouped = (
            self.df.groupby(pd.Grouper(key=self.date_col, freq=self.freq))[self.emb_col]
            .apply(lambda vs: np.vstack(vs))
            .dropna()
            .sort_index()
        )

        idx = grouped.index
        n = len(idx)
        mmd2 = np.full(n, np.nan)

        for i in range(1, n):
            ref = grouped.iloc[i - 1]
            cur = grouped.iloc[i]

            detector = MMDDrift(
                x_ref=ref,
                p_val=self.p_val,
                backend=self.backend,
            )
            pred = detector.predict(cur, return_distance=True)
            mmd2[i] = pred["data"]["distance"]

        return pd.Series(mmd2, index=idx)