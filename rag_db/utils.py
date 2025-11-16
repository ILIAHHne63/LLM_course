# rag_db/utils.py
from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from typing import List, Union

# Модель по умолчанию — возвращает 384-d векторы
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    """
    Обёртка над SentenceTransformer.
    encode(texts) возвращает список списков float (python), не numpy array.
    """
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        texts: str или list[str]
        Возвращает: list[list[float]] (dtype float32 -> python float)
        """
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        # convert_to_numpy=True чтобы можно было контролировать dtype
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embs = embs.astype(np.float32)  # numpy array
        # Преобразуем в обычный список списков (JSON-совместимо)
        result = embs.tolist()
        return result[0] if single else result
