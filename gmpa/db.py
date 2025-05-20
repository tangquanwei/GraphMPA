import faiss
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg as la
from loguru import logger
import tiktoken
import cyberdb
import time
from tqdm.auto import tqdm

from .llm import BaseLLM
from .embed import BaseEmbedder
from .text_splitter import ChineseRecursiveTextSplitter
from .text_splitter import split_text

class CyberDB:
    def __init__(
        self,
        file_name,
        start_server=True,
    ):
        self.db = None
        self.client = None
        self.proxy = None
        if start_server:
            self._start_server(file_name=file_name)

    def _start_server(
        self,
        file_name,
        host="127.0.0.1",
        port=9980,
        password="123456",
    ):
        self.db = cyberdb.Server()
        self.db.set_backup(file_name, cycle=60)
        self.db.start(host=host, port=port, password=password)
        return self.db

    def _get_client(
        self,
        host="127.0.0.1",
        port=9980,
        password="123456",
        times=10,
    ):
        while not self.client and times > 0:
            try:
                self.client = cyberdb.connect(host=host, port=port, password=password)
            except:
                time.sleep(2)
            times -= 1
        return self.client

    def get_proxy(self, times=10):
        if self.client is None:
            self._get_client()

        while not self.proxy and times > 0:
            try:
                self.proxy = self.client.get_proxy()
                self.proxy.connect()
            except:
                time.sleep(2)
            times -= 1

        return self.proxy


class FAISSDB:
    def __init__(
        self,
        embedder: BaseEmbedder,
        tokenizer=None,
        llm: BaseLLM = None,
        use_gpu=False,  # install faiss_gpu_cu121
    ) -> None:
        self.embedder = embedder

        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer
        self.use_gpu = use_gpu

        self.llm = llm
        self.embeddings = None
        self.index = None
        self.context = None  # key
        self.values = None  # value
        self.graph = None
        self.communities = None
        self.stat = {}  # statistics information

    def len_token(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _slipt(
        self,
        context,
        chinese=False,
        chunk_size=100,
        chunk_overlap=0,
    ):
        chunks = None
        if chinese:
            text_splitter = ChineseRecursiveTextSplitter(
                keep_separator="start",
                is_separator_regex=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = text_splitter.split_text(context)
        else:
            assert self.tokenizer is not None, "No tokenizer found :("
            chunks = split_text(
                text=context,
                tokenizer=self.tokenizer,
                max_tokens=chunk_size,
                overlap=chunk_overlap,
            )
        return chunks

    def from_document(
        self,
        context: str,
        chinese: bool = False,
        chunk_size: int = 100,
        chunk_overlap: int = 0,
        normalize_embeddings=False,
        enable_large_chunk_summary: bool = False,
        large_chunk_size: int = 512,
        summary_max_new_tokens: int = 100,
    ):
        assert context != "", "context is empty :("
        chunks = []
        if enable_large_chunk_summary:
            large_chunks = self._slipt(
                context,
                chinese=chinese,
                chunk_size=large_chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for large_chunk in tqdm(large_chunks, desc="Large chunks summary: "):
                summary = self.llm.summarize(
                    context=large_chunk,
                    max_new_tokens=summary_max_new_tokens,
                )
                chunks.append(summary)
            chunks_info = {
                "len": len(chunks),
                "max": max(L := list(map(len, chunks))),
                "avg": round(sum(L) / len(L), 2),
                "min": min(L),
                "token_max": max(T := list(map(self.len_token, chunks))),
                "token_avg": round(sum(T) / len(T), 2),
                "token_min": min(T),
            }
            logger.info("large chunks info:" + str(chunks_info))

        smell_chunks = self._slipt(
            context,
            chinese=chinese,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(smell_chunks)

        self.from_list(chunks, normalize_embeddings=normalize_embeddings)

    def from_list(
        self,
        keys: list[str],
        values: list[str] | None = None,
        normalize_embeddings=False,
    ):
        assert isinstance(keys, list) and len(keys) > 0, "bad keys :("

        # statistical information
        chunks = keys
        chunks_info = {
            "len": len(chunks),
            "max": max(L := list(map(len, chunks))),
            "avg": round(sum(L) / len(L), 2),
            "min": min(L),
            "token_max": max(T := list(map(self.len_token, chunks))),
            "token_avg": round(sum(T) / len(T), 2),
            "token_min": min(T),
        }
        self.stat["chunk"] = chunks_info
        logger.info("chunks info:" + str(chunks_info))

        self.context = np.array(keys)
        self.values = values

        logger.info("encoding")
        self.embeddings = self.embedder.encode(
            keys,
            normalize_embeddings=normalize_embeddings,
        )

        logger.info("indexing")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # declare GPU resources
            self.index = faiss.index_cpu_to_gpu(
                res, 0, self.index
            )  # transfer the index to the GPU
        self.index.add(self.embeddings)
        data_info = {
            "C": {len(self.context)},
            "E": {len(self.embeddings)},
            "I": {self.index.ntotal},
        }
        logger.info(str(data_info))

    def retrive(self, query: str, top_k: int):
        assert self.index is not None, "self.index is None"
        assert self.embeddings is not None, "self.embeddings is None"
        assert self.context is not None, "self.context is None"
        # TODO: add m3
        embedding = self.embedder.encode([query])
        D, I = self.index.search(embedding, top_k)
        result = []
        if self.values is not None:  # 组合values
            for d, i in zip(D.flatten(), I.flatten()):
                result.append((d, self.context[i] + ":" + self.values[i]))
        else:
            for d, i in zip(D.flatten(), I.flatten()):
                result.append((d, self.context[i]))
        return result

    def build_graph_louvain(
        self,
        top_k: int = 10,
        threshold: float = 0.5,
        seed: int = 123,
        init_resolution: float = 1,
    ) -> list[str]:
        """
        Louvain community detection

        init_resolution: float  初始社区划分的分辨率gamma, gamma<1: 倾向于形成更大的社区; gamma>1: 倾向于形成更小的社区

        """

        assert top_k > 0, "top_k should > 0"
        assert 0 < threshold < 1, "threshold should > 0 and <1"
        assert init_resolution > 0, "init_resolution should > 0"
        args = locals()
        logger.info(str(args))

        # build graph
        edges = []

        D, I = self.index.search(self.embeddings, top_k)
        for u, d, i in zip(range(len(D)), D, I):
            for v, w in zip(i[1:].tolist(), d[1:].tolist()):
                if w < threshold:
                    break
                edges.append((u, v, w))

        self.graph = nx.Graph()

        # self.graph.add_nodes_from(range(len(self.embeddings)))
        self.graph.add_weighted_edges_from(edges)
        V = self.graph.number_of_nodes()
        E = self.graph.number_of_edges()
        graph_info = {
            "V": V,
            "E": E,
        }
        self.stat["graph"] = graph_info
        logger.info(str(graph_info))

        # build community
        resolution = init_resolution
        communities = []

        while resolution > 0:
            communities = nx.community.louvain_communities(
                self.graph,
                seed=seed,
                resolution=resolution,
            )
            if len(communities) == 0:
                resolution -= 0.95
                logger.info(f"resolution={resolution} failed")
            else:
                break

        if len(communities) == 0:
            logger.warning("len(communities) == 0")
            return []
        self.communities = communities
        community_info = {
            "len": len(communities),
            "max": max(L := list(map(len, communities))),
            "avg": round(sum(L) / len(L), 2),
            "min": min(L),
        }
        self.stat["community"] = community_info
        logger.info("community: " + str(community_info))

        # Ensure that there is at least one community with more than one node and at least one community
        community_context = list(
            map(lambda x: "\n".join(self.context[[*x]]), communities)
        )

        # stat
        community_context_info = {
            "token_max": max(T := list(map(self.len_token, community_context))),
            "token_avg": round(sum(T) / len(T), 2),
            "token_min": min(T),
        }
        self.stat["community_context"] = community_context_info
        logger.info(f"community_context: " + str(community_context_info))

        return community_context

    def build_graph_leiden(
        self,
        top_k: int = 10,
        threshold: float = 0.5,
        seed: int = 123,
        init_resolution: float | None = None,
        with_wights: bool = False,
    ) -> list[str]:
        if init_resolution is not None:
            logger.info("init_resolution is not None")
        self.graph = ig.Graph()
        edges = []
        attr = {"weight": []}
        D, I = self.index.search(self.embeddings, top_k)
        for u, d, i in zip(range(len(D)), D, I):
            for v, w in zip(i[1:].tolist(), d[1:].tolist()):
                if w == 1:  # itself
                    continue
                if w < threshold:
                    break
                edges.append((u, v))
                attr["weight"].append(w)

        self.graph.add_vertices(V := len(self.embeddings))
        self.graph.add_edges(edges, attributes=attr if with_wights else None)

        logger.info(f"G(V, E) = ({V}, {len(edges)})")

        communities = la.find_partition(
            graph=self.graph,
            partition_type=la.ModularityVertexPartition,
            seed=seed,
            weights=attr["weight"] if with_wights else None,
        )
        l, m, n = (
            len(communities),
            max(map(len, communities)),
            min(map(len, communities)),
        )

        # Ensure that there is at least one community with more than one node and at least one community
        logger.info(f"communities l, m, n= {l},{m},{n}")

        community_context = []
        for i in communities:
            if len(i) > 1:
                community_context.append("\n".join(self.context[[*i]]))

        l, m, n = (
            len(community_context),
            max(map(len, community_context)),
            min(map(len, community_context)),
        )
        logger.info(f"community_context l, m, n= {l},{m},{n}")

        return community_context

    def get_report(self):
        """
        get statistics of the dataset
        """
        return self.stat
