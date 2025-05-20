from abc import ABC, abstractmethod
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from loguru import logger
import gc
# from utils import get_device



class BaseLLM(ABC):
    @abstractmethod
    def summarize(self, context: str, max_new_tokens: int) -> str:
        pass

    @abstractmethod
    def answer_question(self, question: str, max_new_tokens: int) -> str:
        pass

    @abstractmethod
    def reload_model(self):
        pass


class HF_LLM(BaseLLM):

    def __init__(
        self,
        model_path: str,
        device_map: str = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        dtype: str = None,
        attn_impl: str = None,
    ):
        if model_path:
            self.model_path = model_path
            self.device_map = device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype or torch.bfloat16,
                device_map=device_map ,
                attn_implementation=attn_impl,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.device = self.model.device

    def _generate(self, prompt, max_new_tokens):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        res = self.model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
        )
        return self.tokenizer.decode(
            res[0][len(input_ids[0]) :],
            skip_special_tokens=True,
        )

    def answer_question(
        self,
        context,
        max_new_tokens=256,
        system_prompt=None,
        verbose=False,
    ):
        assert context is not None and len(context) > 0, "len(context)==0"
        message = [
            {"role": "user", "content": context},
        ]
        if system_prompt:
            message.insert(0, {"role": "system", "content": system_prompt})
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        if verbose:
            logger.info(f"prompt: {prompt}")

        return self._generate(prompt, max_new_tokens)

    def summarize(self, context: str, max_new_tokens=512):
        assert context is not None and len(context) > 0, "len(context)==0"

        prompt = f"Write a summary of the following, including as many key details as possible: {context}:"
        text = self._generate(prompt, max_new_tokens)
        text = text.strip()
        return text

    def reload_model(self):
        logger.info("reloading model ...")

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            attn_implementation="flash_attention_2",
        )

        logger.info("reloaded model done!")


class MistralLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        device_map="cuda:0",
    ):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_inference.transformer import Transformer

        logger.info(f"loading Mistral from {model_path}, device: {device_map}")
        self.tokenizer = MistralTokenizer.from_file(f"{model_path}/tekken.json")
        self.model = Transformer.from_folder(
            model_path,
            dtype=torch.float16,
            device=device_map,
        )
        self.model_path = model_path
        self.device_map = device_map

    def reload_model(self):
        from mistral_inference.transformer import Transformer

        logger.info("reloading model ...")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = Transformer.from_folder(
            self.model_path,
            dtype=torch.float16,
            device=self.device_map,
        )
        logger.info("reloaded model done!")

    def answer_question(self, context, max_new_tokens=256):
        from mistral_inference.generate import generate
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        assert context is not None and len(context) > 0, "len(context)==0"
        if len(context) > 10000:
            context = context[:10000]
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=context)]
        )

        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        result = ""
        try:
            out_tokens, _ = generate(
                [tokens],
                self.model,
                max_tokens=max_new_tokens,
                temperature=0,
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        except Exception as e:
            logger.info(e)
            if "CUDA out of memory" in str(e):
                self.reload_model()

        return result

    def summarize(self, context: str, max_new_tokens=512):
        assert context is not None and len(context) > 0, "len(context)==0"

        prompt = f"Write a summary of the following, including as many key details as possible: {context}:"
        result = self.answer_question(prompt, max_new_tokens)
        text = result.strip()
        return text
