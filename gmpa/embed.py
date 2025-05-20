from functools import cache
from loguru import logger
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from loguru import logger


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, text, normalize_embeddings):
        pass


class STEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        device_map="cuda:0",
    ):
        self.model = SentenceTransformer(model_name, device=device_map)
        logger.info("load st_embedder done!")

    def encode(self, text, normalize_embeddings=False,**kwargs):
        return self.model.encode(text, normalize_embeddings=normalize_embeddings,**kwargs)


class BgeM3Embedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        device_map="cuda:0",
    ):
        # from FlagEmbedding import BGEM3FlagModel
        # self.model = BGEM3FlagModel(model_name, use_fp16=True, device=device_map)
        logger.info("load m3 embedder done!")

    def encode(self, text, normalize_embeddings=False):
        assert normalize_embeddings == False
        embed = self.model.encode(text)["dense_vecs"]
        if normalize_embeddings:
            # embed = embed / np.linalg.norm(embed, ord=2, axis=1, keepdims=True)
            raise NotImplementedError
        return embed
