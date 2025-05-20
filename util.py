import jsonlines
import pyarrow.parquet as pq
from functools import cache
import re
import subprocess
import json
import faiss
import numpy as np
from loguru import logger
import os
import importlib.util
import sys

from gmpa.llm import BaseLLM, HF_LLM, MistralLLM
from gmpa.embed import STEmbedder
import statistics


def get_dataset(a):
    print(a)


def get_ipv4_address(interface: str = "enp4s0") -> str:
    try:
        result = subprocess.run(
            ["ip", "address", "show", interface],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/", result.stdout)
        if match:
            ipv4_address = match.group(1)
            return ipv4_address
        else:
            return "No IPv4 address found for the specified interface."
    except subprocess.CalledProcessError as e:
        return f"An error occurred while trying to get the IP address: {e}"


ip = get_ipv4_address()
prefix = None
if ip.endswith("142"):
    prefix = "/data/qwtang"
    llm_prefix = "/data/LLM"
elif ip.endswith("156"):
    prefix = "/mnt/sda/qwtang"
    llm_prefix = prefix

print(f"run on {ip}, {prefix}")

DATA_PATH: dict[str, dict[str, str]] = {
    "qasper": {
        "test": f"{prefix}/qasper/test.parquet",
        "val": f"{prefix}/qasper/val.parquet",
        "train": f"{prefix}/qasper/train.parquet",
    },
    "quality": {
        "dev": f"{prefix}/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.dev",
        "test": f"{prefix}/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.test",
        "train": f"{prefix}/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.train",
    },
    "narrativeqa": {
        "qaps": f"{prefix}/narrativeqa/qaps.csv",
        "summaries": f"{prefix}/narrativeqa/third_party/wikipedia/summaries.csv",
        "documents": f"{prefix}/narrativeqa/documents.csv",
    },
    "riddlesense": {
        "train": f"{prefix}/riddle_sense/data/train-00000-of-00001-4fd13466bedac037.parquet",
        "val": f"{prefix}/riddle_sense/data/validation-00000-of-00001-5ec8f4b8449d47bf.parquet",
        "test": f"{prefix}/riddle_sense/data/test-00000-of-00001-bd025d2cce659c20.parquet",
    },
    "openbookqa": {
        "train": f"{prefix}/openbookqa/main/train-00000-of-00001.parquet",
        "test": f"{prefix}/openbookqa/main/test-00000-of-00001.parquet",
        "val": f"{prefix}/openbookqa/main/validation-00000-of-00001.parquet",
    },
    "pubmed_qa": {
        "train": f"{prefix}/pubmed_qa/pqaa_train_set.json",
        "val": f"{prefix}/pubmed_qa/pqaa_dev_set.json",
        "test": f"{prefix}/pubmed_qa/pqal_test_set.json",
    },
}

MODEL_PATH: dict[str, str] = {
    # embedding
    "sbert": f"{llm_prefix}/multi-qa-mpnet-base-cos-v1",
    "bge_zh": f"{llm_prefix}/bge-large-zh-v1.5",
    "minilm": f"{llm_prefix}/all-MiniLM-L6-v2",
    "contriever": f"{llm_prefix}/contriever",
    "bge_m3": f"{llm_prefix}/bge-m3",
    # raranker
    "bge_reranker": f"{llm_prefix}/bge-reranker-large",
    "bge_reranker_v2": f"{llm_prefix}/bge-reranker-v2-gemma",
    # llm
    "llama_1b": f"{llm_prefix}/Llama-3.2-1B-Instruct",
    "llama_3b": f"{llm_prefix}/Llama-3.2-1B-Instruct",
    "phi_3b": f"{llm_prefix}/Phi-3.5-mini-instruct",
    "qwen_7b": f"{llm_prefix}/Qwen2.5-7B-Instruct",
    "llama_8b": f"{llm_prefix}/Meta-Llama-3.1-8B-Instruct",
    "1b": f"{llm_prefix}/Llama-3.2-1B-Instruct",
    "3b": f"{llm_prefix}/Llama-3.2-1B-Instruct",
    "8b": f"{llm_prefix}/Meta-Llama-3.1-8B-Instruct",
    "mistral_8b": f"{llm_prefix}/Ministral-8B-Instruct-2410",
}


@cache
def get_quality():
    test_file = DATA_PATH["quality"]["train"]
    test_data = list(jsonlines.open(test_file, "r").iter())
    return test_data


@cache
def get_quality_text_qoa(split="train"):
    test_file = f"/data/qwtang/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.{split}"
    test_data = list(jsonlines.open(test_file, "r").iter())

    def get_qoa(q):
        question = q["question"]
        options = "\n".join([f"{i+1}. {o}" for i, o in enumerate(q["options"])])
        answer = q["gold_label"]
        return question, options, answer

    def get_tq(data):
        text = data["article"]
        questions = data["questions"]
        qoa = list(map(get_qoa, questions))
        return text, qoa

    return list(map(get_tq, test_data))


@cache
def get_qasper_text_questions_answers(split="test"):
    test_file = DATA_PATH["qasper"][split]
    test_data = pq.read_table(test_file).to_pylist()

    def handle(data: dict):
        # text
        texts = []
        texts.append(data.get("title", ""))
        texts.append(data.get("abstract", ""))
        p = data["full_text"]["paragraphs"]
        s = data["full_text"]["section_name"]
        for i, j in zip(s, p):
            texts.append(i if i else "")
            texts.append("\n".join(j))
        try:
            text = "\n\n".join(texts)
        except:
            print(texts)
        # qa
        questions = data["qas"]["question"]
        answers = []
        for as_ in data["qas"]["answers"]:
            answer = []
            for x in as_["answer"]:
                if len(x["extractive_spans"]) != 0:
                    answer.extend(x["extractive_spans"])
                if x["free_form_answer"] != "":
                    answer.append(x["free_form_answer"])
            answer = list(set(map(str.strip, answer)))
            answers.append(answer)
        return text, questions, answers

    return list(map(handle, test_data))


def get_riddlesense_docs_questions_answers():
    # docs from train
    path = DATA_PATH["riddlesense"]["train"]
    dataset = pq.read_table(path).to_pylist()
    template = "{question} Answer: {answer}."
    pattern = re.compile(r"\b([A-E]):\s*(\w+)")

    def handle(data):
        text = data["input"]
        matches = pattern.findall(text)
        matches = dict(matches)
        answer = matches[data["output"]]

        question = text.split("Choices:")[0].strip()
        if question[0] == ".":
            question = question[1:].strip()
        if question[-1] != "?":
            question = question + " ?"
        doc = template.format(question=question, answer=answer)
        return doc

    docs = list(map(handle, dataset))

    # val as test
    path = DATA_PATH["riddlesense"]["val"]
    test_dataset = pq.read_table(path).to_pylist()
    questions = list(map(lambda x: x["input"], test_dataset))
    answers = list(map(lambda x: x["output"], test_dataset))

    return docs, questions, answers


def get_openbookqa_docs_questions_choices_answers():
    # docs from train
    path = DATA_PATH["openbookqa"]["train"]
    dataset = pq.read_table(path).to_pylist()

    def hanle(data):
        """
        question w/o choices
        """
        question = data["question_stem"]
        label = data["choices"]["label"].index(data["answerKey"])
        answer = data["choices"]["text"][label]
        result = f"{question} {answer}."
        return result

    docs = list(map(hanle, dataset))

    path = DATA_PATH["openbookqa"]["test"]
    test_dataset = pq.read_table(path).to_pylist()
    questions = list(map(lambda x: x["question_stem"], test_dataset))
    choices = list(map(lambda x: x["choices"], test_dataset))
    answers = list(map(lambda x: x["answerKey"], test_dataset))
    return docs, questions, choices, answers


def get_pubmed_context_qa():
    """
    pubmed_qa
    """
    import json

    train_path = f"{prefix}/pubmed_qa/pqal/pqal_train_dev_set.json"
    test_path = f"{prefix}/pubmed_qa/pqal/pqal_test_set.json"
    train_data = json.load(open(train_path))
    test_data = json.load(open(test_path))
    docs = []
    dataset = []
    for i, j in train_data.items():
        context = j["CONTEXTS"]
        docs.extend(context)
    for i, j in test_data.items():
        question = j["QUESTION"]
        context = j["CONTEXTS"]
        reason = j["LONG_ANSWER"]
        answers = j["final_decision"]
        docs.extend(context)
        dataset.append(
            {
                "id": i,
                "question": question,
                "answers": answers,
                "context": context,
                "reason": reason,
            }
        )
    return docs, dataset


def get_med_qa_docs_qa(embedder):
    def trans(data):
        #         return f"""{data['question']}
        # options: {json.dumps(data['options'])}
        # ##Answer: {data['answer_idx']}: {data['answer']}"""
        return f"""{data['question']}\n##Answer: {data['answer']}"""

    test_path = f"{prefix}/med_qa/data_clean/questions/US/test.jsonl"
    test_dataset = list(jsonlines.open(test_path, "r").iter())
    train_path = f"{prefix}/med_qa/data_clean/questions/US/train.jsonl"
    train_cache_path = f"{prefix}/med_qa/train_cache.jsonl"
    if os.path.exists(f"{prefix}/med_qa/train_cache.jsonl"):
        return list(jsonlines.open(train_cache_path, "r").iter()), test_dataset
    else:
        train_dataset = list(jsonlines.open(train_path, "r").iter())
        docs = list(map(trans, train_dataset))
        embeddings = embedder.encode(docs)
        questions = list(map(lambda data: data["question"], test_dataset))
        question_embeddings = embedder.encode(questions)
        index = faiss.IndexFlatIP(1024)
        index.add(embeddings)
        D, I = index.search(question_embeddings, 5)
        docs_np = np.array(docs)
        new_docs = docs_np[I.flatten()]
        new_docs_unique = np.unique(new_docs)
        print(len(new_docs_unique))
        docs_list = new_docs_unique.tolist()
        jsonlines.open(f"{prefix}/med_qa/train_cache.jsonl", "w").write_all(docs_list)
    return docs_list, test_dataset


def get_medmcqa_docs_qa(embedder):
    def trans(data):
        answer_idx = ["A", "B", "C", "D"][data["cop"]]
        answer = data[["opa", "opb", "opc", "opd"][data["cop"]]]
        docs = f"""Question: {data['question']}
        Options: (A) {data['opa']}, (B) {data['opb']}, (C) {data['opc']}, (D) {data['opd']}
        Answer: ({answer_idx}) {answer}
        """
        return docs

    train_path = f"{prefix}/medmcqa/data/train-00000-of-00001.parquet"
    test_path = f"{prefix}/medmcqa/data/validation-00000-of-00001.parquet"
    test_dataset = pq.read_table(test_path).to_pylist()
    train_cache_path = f"{prefix}/medmcqa/train_cache.jsonl"
    if os.path.exists(f"{prefix}/medmcqa/train_cache.jsonl"):
        logger.info("Load docs from cache")
        return list(jsonlines.open(train_cache_path, "r").iter()), test_dataset
    else:
        train_dataset = pq.read_table(train_path).to_pylist()
        logger.info(f"compute docs size: {len(train_dataset)}")
        docs = list(map(trans, train_dataset))

        logger.info(f"embedding docs...")
        embeddings = embedder.encode(docs)
        questions = list(map(lambda data: data["question"], test_dataset))

        logger.info(f"embedding questions ...")
        question_embeddings = embedder.encode(questions)

        logger.info(f"indexing ...")
        index = faiss.IndexFlatIP(1024)
        index.add(embeddings)
        D, I = index.search(question_embeddings, 5)
        docs_np = np.array(docs)
        new_docs = docs_np[I.flatten()]
        new_docs_unique = np.unique(new_docs)
        logger.info(len(new_docs_unique))
        docs_list = new_docs_unique.tolist()
        jsonlines.open(train_cache_path, "w").write_all(docs_list)
    return docs_list, test_dataset


### MIRAGE BEGIN


class QADataset:

    def __init__(self, data, dir="."):
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(os.path.join(dir, "benchmark.json")))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if type(key) == int:
            return self.dataset[self.index[key]]
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")


def locate_answer(sentence: str):

    ans = re.findall("^\s*(A|B|C|D)$", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D) or", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D) and", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)/", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D),", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall(":\s*(A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\.", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall('^\s*(A|B|C|D)"', sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D):", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    return "A"


def locate_answer4pub_llama(sentence: str):

    sentence = sentence.split("Answer:")[-1]

    ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall("OPTION (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall('^\s*(A|B|C|D)"', sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D):", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    return "A"


def evaluate(dataset, save_dir, split="test", locate_fun=locate_answer):

    flag = False
    pred = []
    empty_count = 0
    na_count = 0
    answer_list = ["A", "B", "C", "D"]
    answer2idx = {ans: i for i, ans in enumerate(answer_list)}

    total_len = len(dataset)

    # for i, fpath in enumerate(sorted([f for f in os.listdir(save_dir) if f.endswith(".json")])[:total_len]):
    for q_idx in range(len(dataset)):
        fpath = os.path.join(save_dir, split + "_" + dataset.index[q_idx] + ".json")
        answers = []
        for it in json.load(open(fpath))[:1]:
            answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
        # answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
        answers = [ans for ans in answers if ans != "NA"]
        if len(answers) == 0:
            pred.append(-1)
            continue
        ans = statistics.mode(answers)
        if ans in answer_list:
            pred.append(answer_list.index(ans))
        else:
            pred.append(-1)

    truth = [answer2idx[item["answer"]] for item in dataset]
    if len(pred) < len(truth):
        truth = truth[: len(pred)]
        flag = True

    acc = (np.array(truth) == np.array(pred)).mean()
    std = np.sqrt(acc * (1 - acc) / len(truth))
    return acc, std, flag


### MIRAGE END

get_func = {
    "get_qasper_text_questions_answers": get_qasper_text_questions_answers,
    "get_riddlesense_docs_questions_answers": get_riddlesense_docs_questions_answers,
    "get_openbookqa_docs_questions_choices_answers": get_openbookqa_docs_questions_choices_answers,
    "get_pubmed_context_qa": get_pubmed_context_qa,
    "get_quality": get_quality,
    "get_med_qa_docs_qa": get_med_qa_docs_qa,
    "get_medmcqa_docs_qa": get_medmcqa_docs_qa,
}


def load_python_file(file_path):
    """
    动态加载系统中指定路径的 Python 文件，并执行其内容。

    :param file_path: 需要加载的 `.py` 文件的绝对路径
    :return: 模块对象
    """
    module_name = file_path.split("/")[-1].replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


get_llm: dict[str, BaseLLM] = {
    "1b": lambda device_map="auto": HF_LLM(MODEL_PATH["llama_1b"], device_map),
    "3b": lambda device_map="auto": HF_LLM(MODEL_PATH["llama_3b"], device_map),
    "7b": lambda device_map="auto": HF_LLM(MODEL_PATH["qwen_7b"], device_map),
    "8b": lambda device_map="auto": HF_LLM(MODEL_PATH["llama_8b"], device_map),
    "mistral_8b": lambda device_map="cuda": MistralLLM(
        MODEL_PATH["mistral_8b"], device_map
    ),
    "qwen_7b": lambda device_map="auto": HF_LLM(MODEL_PATH["qwen_7b"], device_map),
    "hf": HF_LLM,
}


get_embedder = {
    "bge": lambda model_path=MODEL_PATH["bge_zh"], device_map="cuda:0": STEmbedder(
        model_path,
        device_map,
    ),
    "sbert": lambda model_path=MODEL_PATH["sbert"], device_map="cuda:0": STEmbedder(
        model_path,
        device_map,
    ),
    "minilm": lambda model_path=MODEL_PATH["minilm"], device_map="cuda:0": STEmbedder(
        model_path,
        device_map,
    ),
    "m3": lambda model_path=MODEL_PATH["bge_m3"], device_map="cuda:0": STEmbedder(
        model_path,
        device_map,
    ),
}
