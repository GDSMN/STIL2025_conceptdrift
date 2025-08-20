try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import numpy as np
from typing import Optional


class vectors:
    """
    Adiciona uma coluna de embeddings ao dataframe.
    model_name: str
    model_kwargs: dict
    encode_kwargs: dict
    """

    def __init__(
        self,
        model_name: str = "neuralmind/bert-large-portuguese-cased",
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {"device": "cuda"}
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": False}

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )

    def attach(self, df: pd.DataFrame, text_col: str, out_col: str = "embeddings") -> pd.DataFrame:
        df_out = df.copy()
        textos = df_out[text_col].fillna("").astype(str).tolist()
        mat = np.asarray(self.embedder.embed_documents(textos), dtype=np.float32)
        df_out[out_col] = list(mat)
        return df_out