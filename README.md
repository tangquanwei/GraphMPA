# GraphMPA

## Overview

The official implementation of GraphMPA: 

A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment

Traditional RAG face challenges in achieving global understanding and aligning responses with human ethical and quality preferences. To address these issues, we propose GraphMPA, a comprehensive graph-based framework with mode-seeking preference alignment. Our approach constructs a hierarchical document graph using a general similarity measurement, mimicking human cognitive processes for information understanding and synthesis.

![graphMPA](img/graphmpa.png)


## Installation

Before using GraphMPA, ensure Python 3.11+ is installed. 

### Clone the GraphMPA repository:

```bash
git clone https://github.com/tangquanwei/GraphMPA.git
cd raptor
```

### Install necessary dependencies:

Using conda:
```bash
conda create --name graph_mpa python=3.11 -y
conda activate graph_mpa
```

Or using pip:   

```bash
pip install -r requirements.txt
```

## Basic Usage

To get started with GraphMPA, follow these steps:

### Setting Up 

Download model, recommend is here:

LLM: `Qwen/Qwen2.5-7B-Instruct`

Embedding: `BAAI/bge-m3`

### Load model 

Load Embedding model:

```python
from gmpa.embed import STEmbedder
model_path = 'BAAI/bge-m3'
device_map = 'auto'
embedder = STEmbedder(model_path, device_map)
```

Load Language model:

```python
from gmpa.llm import HF_LLM
model_path = 'Qwen/Qwen2.5-7B-Instruct'
device_map = 'auto'
llm = HF_LLM(model_path, device_map)
```

Build RAG Object:

```python
from gmpa import Rag
rag = Rag(embedder, llm)
```

Load Data:

```python
path = "data/The Great Gatsby.txt"
with open(path, encoding='latin-1') as f:
    text = f.read()
len(text)
```

Construct Database:

```Python
rag.build(
    document=text,
    chinese=True,
    enable_large_chunk_summary=True,
)
```

Sample Question Retrieve:

```Python
question="What is the main content of this book?"
context=rag.retrive(question)
print(context)
```

Make a prompt:

```Python
promplate="""Given the Question, Context below, provide a logical reasoning to get the answer. Please use the format of: ##Reason: <reason> ##Answer: <answer>.

Question: {question}

Context: {context}"""
prompt=promplate.format(question=question, context=context)
print(prompt)
```

Answer Question:

```Python
answer=llm.answer_question(prompt)
print(answer)
```

## Contributing

GraphMPA is an open-source project, and contributions are warmly welcomed. Whether you're fixing bugs, implementing new features, or enhancing the documentation, your efforts are greatly appreciated.

## Citing
```
@inproceedings{tang-etal-2025-comprehensive,
    title = "A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment",
    author = "Tang, Quanwei  and
      Lee, Sophia Yat Mei  and
      Wu, Junshuang  and
      Zhang, Dong  and
      Li, Shoushan  and
      Cambria, Erik  and
      Zhou, Guodong",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1108/",
    doi = "10.18653/v1/2025.findings-acl.1108",
    pages = "21504--21523",
    ISBN = "979-8-89176-256-5",
    abstract = "Recent advancements in retrieval-augmented generation (RAG) have enhanced large language models in question answering by integrating external knowledge. However, challenges persist in achieving global understanding and aligning responses with human ethical and quality preferences. To address these issues, we propose GraphMPA, a comprehensive graph-based framework with mode-seeking preference alignment. Our approach constructs a hierarchical document graph using a general similarity measurement, mimicking human cognitive processes for information understanding and synthesis. Additionally, we introduce mode-seeking preference optimization to better align model outputs with human preferences through probability-matching constraints. Extensive experiments on six datasets demonstrate the effectiveness of our GraphMPA."
}
```

If you find our paper or this repository helpful, please consider citing our work – it’s much appreciated! 😊
