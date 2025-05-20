from typing import Literal
from tqdm import tqdm
import gc
from loguru import logger
import pickle
import os

from .db import FAISSDB
from .llm import BaseLLM
from .embed import BaseEmbedder


class Rag:
    def __init__(
        self,
        embedder: BaseEmbedder,
        qa_llm: BaseLLM,
        summary_llm: BaseLLM = None,
    ):
        self.embedder = embedder
        self.qa_llm: BaseLLM = qa_llm
        self.summary_llm: BaseLLM = summary_llm if summary_llm else qa_llm
        self.layers: list[FAISSDB] = []
        self.num_layers: int | None = None
        self.summary_cache: dict = None

    def retrive(
        self,
        query: str,
        top_k: int = 5,
        use_k: int = 5,
        strategy: str = "topk",
        sort_by_embedding=False,
        from_layer=False
    ):
        results = []
        if from_layer:
            for i in range(len(self.layers)):
                layer=self.layers[i]
                result =layer.retrive(query, top_k)
                results.extend([(r,i) for r in result])
        else:
            for layer in self.layers:
                result = layer.retrive(query, top_k)
                results.extend(result)
            if sort_by_embedding:
                results.sort(key=lambda x: x[0])
                results = list(map(lambda x: x[1], results))
                results = results[:top_k]
            else:
                results = list(map(lambda x: x[1], results))
            return results[: min(use_k, top_k)]

    def build(
        self,
        document: str | list[str],
        community_algorithm: Literal["louvain", "leiden"] = "leiden",
        top_k: int = 10,
        num_layers: int = 2,
        threshold: float = 0.5,
        summarization_length: int = 256,
        seed: int = 123,
        init_resolution: float = 1,
        normalize_embeddings=False,
        chunk_size: int = 100,
        chunk_overlap: int = 0,
        chinese: bool = False,
        cache_file_name: str | None = None,  # dataset name
        enable_large_chunk_summary: bool = False,
        large_chunk_size: int = 512,
    ):
        # ues cache
        if cache_file_name:
            # eg. "output/cache/{dataset}.pkl"
            try:
                with open(cache_file_name, "rb") as f:
                    self.summary_cache = pickle.load(f)
                logger.info(f"load from {cache_file_name}")
            except Exception as e:
                logger.warning(f"pickle.load(f) [X]\n{e}")
                if not os.path.exists(_path := cache_file_name.rsplit("/", 1)[0]):
                    os.mkdir(_path)
                    logger.info(f"os.mkdir({_path}) [V]")

                self.summary_cache = {}
                logger.info("self.summary_cache = {} [V]")

        for i in tqdm(range(num_layers), desc="Build layers"):
            # step 1 build db
            db = FAISSDB(self.embedder, llm=self.summary_llm)

            if isinstance(document, str):
                db.from_document(
                    document,
                    normalize_embeddings=normalize_embeddings,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chinese=chinese,
                    enable_large_chunk_summary=enable_large_chunk_summary,
                    large_chunk_size=large_chunk_size,
                    summary_max_new_tokens=summarization_length,
                )
            elif isinstance(document, list):
                db.from_list(document, normalize_embeddings=normalize_embeddings)
            else:
                raise ValueError("document must be a string or a list of strings")

            self.layers.append(db)

            if num_layers - i == 1:
                continue

            # # step 2 make summary
            community_context = []

            if community_algorithm == "louvain":
                community_context = db.build_graph_louvain(
                    top_k=top_k,
                    threshold=threshold,
                    seed=seed,
                    init_resolution=init_resolution,
                )
            elif community_algorithm == "leiden":
                community_context = db.build_graph_leiden(
                    top_k=top_k,
                    threshold=threshold,
                    seed=seed,
                    init_resolution=init_resolution,
                )
            else:
                raise ValueError("community_algorithm must be either leiden or louvain")

            if community_context == []:
                logger.warning("No community context found")
                continue

            summarization_outputs = []
            model = self.summary_llm if self.summary_llm else self.qa_llm
            for text in tqdm(community_context, desc="summarizing"):
                output = None
                if self.summary_cache and text in self.summary_cache:
                    output = self.summary_cache.get(text)
                else:
                    try:
                        output = model.summarize(
                            text,
                            max_new_tokens=summarization_length,
                        )
                        if self.summary_cache is not None:
                            self.summary_cache[text] = output
                    except Exception as e:
                        logger.warning(f"summarization failed {e}, L:" + str(len(text)))
                        model.reload_model()
                        continue

                summarization_outputs.append(output)
                # print(output)
            l, m, n = (
                len(summarization_outputs),
                max(map(len, summarization_outputs)),
                min(map(len, summarization_outputs)),
            )
            logger.info(f"summarization_outputs l, m, n= {l},{m},{n}")
            document = summarization_outputs

        if self.summary_cache:
            with open(cache_file_name, "wb") as f:
                pickle.dump(self.summary_cache, f)
            logger.info("dump summary_cache len=" + str(len(self.summary_cache)))

    def clear(self):
        self.layers = []
        self.num_layers = None
        gc.collect()

    def get_stat(self):
        stats = []
        for i, db in enumerate(self.layers):
            stat = db.get_report()
            stats.append(stat)
        return stats
