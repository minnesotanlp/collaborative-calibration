"""Class for base agents"""
import logging
import os
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss

from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.cohere import Cohere
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from .memory import BaseAgentMemory
from simulation_utils import TEST_API_MODELS, TEST_HF_MODELS, get_model_specific_chat_template
from load_pretrained import causal_lm_generate

TEST_INTERNAL_MEM_PATH = Path("../data/memory/")
DEFAULT_MODEL_TYPE = "gpt-3.5-turbo-1106"

DEFAULT_RATING_PROMPT = (
    " rate the level of ambiguity in the input query (a float from 0 to 1);"
    " rate the level of complexity of the input query (a float from 0 to 1);"
    " rate your level of ability for solving the input query (a float from 0 to 1);"
    " Note that your uncertainty on the correctness of your answer is affected by input ambiguity, task complexity, and your own knowledge and abilities."
    " Based on this,"
)


class BaseAgent:
    def __init__(
        self,
        id: str,
        persona_description: Optional[str] = "",
        adversarial: Optional[bool] = False,
        model_temperature: Optional[float] = 1.0,
        model_type: Optional[str] = DEFAULT_MODEL_TYPE,
        model_wrapper: Optional[str] = None,
        model_token_limit: Optional[int] = 1024,
        model_stops: Optional[List[str]] = [],
        pretrained: Optional[Tuple[Any, Any]] = None,
        reflecting: Optional[bool] = False,
        detailed_rating: Optional[bool] = True,
    ):
        self.id = id
        self.persona = persona_description
        self.adversarial = adversarial
        self.model_type = model_type
        self.temperature = model_temperature
        self.token_limit = model_token_limit
        self.model_stops = model_stops
        if self.model_type in TEST_API_MODELS["openai-chat"]:
            self.model_wrapper = model_wrapper or ChatOpenAI(
                model=self.model_type,
                temperature=self.temperature,
                max_tokens=self.token_limit,
                max_retries=50,
                model_kwargs={"stop": self.model_stops},
                request_timeout=800,
            )
        elif self.model_type in TEST_API_MODELS["cohere"]:
            # wrap Cohere-Generate
            self.model_wrapper = model_wrapper or Cohere(
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                temperature=self.temperature,
                max_tokens=self.token_limit,
                max_retries=50,
                stop=self.model_stops,
            )
        elif self.model_type.split("/")[-1] in itertools.chain(*TEST_HF_MODELS.values()):
            # loaded model instance from vllm/HF
            self.pretrained = pretrained
        else:
            raise NotImplementedError
        self.message_history: Dict[str, List[str]] = {"court_0": []}
        self.reflecting = reflecting

        # local memory: currently includes buffer and summary (high-level reflection)
        self.memory = BaseAgentMemory(
            llm=ChatOpenAI(temperature=0.9, max_tokens=1024, max_retries=50, request_timeout=800),
            memory_retriever=self.get_memory_retriever(docs_retrieved=5),
            reference=self.id,
            reflecting=self.reflecting,
        )
        self.callback_handlers = (
            [OpenAICallbackHandler()] if self.model_type in TEST_API_MODELS["openai-chat"] else None
        )
        self.detailed_rating = detailed_rating
        self.default_trigger_prompt = (
            "State your answer (as short as possible, in one or a few words), then"
            f"{DEFAULT_RATING_PROMPT if self.detailed_rating else ''}"
            " give a float (between 0 to 1) indicating your overall confidence on how likely that your answer is correct."                
            " Follow this format:\nAnswer:<answer>\nAmbiguity:<ambiguity score>\nComplexity:<complexity score>\nAbility:<ability score>\nConfidence:<confidence>"
        )

    def get_model_wrapper(self) -> Any:
        return self.model_wrapper

    def chain(self, prompt_template: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.model_wrapper,
            prompt=prompt_template,
            verbose=True,
            callbacks=self.callback_handlers
        )

    def reset_message_history(self):
        """Reset message history"""
        self.message_history = {k: [] for k in self.message_history}

    def write_message_history(self, speaker: str, message: str, specified_id: str):
        """Add current message into message history."""
        if specified_id in self.message_history:
            self.message_history[specified_id].append(f"{speaker}: {message}")
        else:
            self.message_history[specified_id] = [f"{speaker}: {message}"]

    def get_memory_retriever(self, docs_retrieved: int):
        """Add retriever based on embedding similarity and recency.
        Using default model OpenAI Ada v2, with default embedding dimension 1536.
        """
        embeddings_model: OpenAIEmbeddings = OpenAIEmbeddings()
        index: int = faiss.IndexFlatL2(1536)
        return TimeWeightedVectorStoreRetriever(
            vectorstore=FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}),
            k=docs_retrieved,
        )

    def self_deliberate(self, query):
        """Simple zero-shot prompting"""
        sys_msg = SystemMessage(content="Consider the following question carefully.")
        human_msg = HumanMessagePromptTemplate.from_template(
            template= f"Question:{query}\n\n{self.default_trigger_prompt}"
        )
        res = self.chain(ChatPromptTemplate.from_messages([sys_msg, human_msg])).predict(callbacks=self.callback_handlers).strip()
        return res

    def self_deliberate_with_pretrained_instance(self, query: str, use_hf_chat_template: bool = False):
        """Simple zero-shot prompting"""
        user_msg = (
            "Consider the following question carefully.\n"
            f"Question:{query}\n\n{self.default_trigger_prompt}"
        )
        if use_hf_chat_template:
            chat_prompt = [
                {"role": "user", "content": "You are a helpful AI agent."},
                {"role": "assistant", "content": "Hi! I will be helpful, factual and truthful."},
                {"role": "user", "content": user_msg},
            ]
        else:
            chat_prompt = get_model_specific_chat_template(self.model_type, user_msg)

        prob, res = causal_lm_generate(
            self.pretrained[0], self.pretrained[1], chat_prompt, return_joint_prob=True
        )
        logging.debug(f"self_deliberate-{self.id}; {res}; model prob: {prob}")
        return prob, res

    def generate_self_evaluation(
        self,
        task: str,
        assigned_stance: str,
    ) -> List[Tuple[str, float]]:
        """For long-form generation, get self-evaluation on the quality of the generation"""
        prompt = f"""
For the task: '{task}'\nHere is your previous response: '{assigned_stance}'
Evaluate your response by generating the following:
List related premises/assumptions you had in the response and your associated confidences (float, from 0 to 1),
following this format: Premise:<a premise>Confidence:<confidence>.
Then, determine whether the task is ambiguous. If so, output "[Ambiguous]", and list several possible interpretations,
following this format: Interpretation:<interpretation>. Mark the interpretation that matches your response with an asterisk (*) at the end of the line.
"""
        res = (
            self.chain(
                ChatPromptTemplate.from_messages(
                    [HumanMessagePromptTemplate.from_template(template=prompt)]
                )
            )
            .predict(
                callbacks=self.callback_handlers,
            )
            .strip()
        )
        logging.debug(f"generate_self_evaluation ({self.model_type}) {prompt}\n\n{res}")
        argument = res.replace("Evaluation:", "").strip()
        return argument
    

    def generate_argument(
        self,
        q: str,
        assigned_stance: str,
    ) -> str:
        prompt = f"""
You are participating in a debate on the question: '{q}'\nYour assigned stance on the question is '{assigned_stance}'
Generate some arguments/evidence (no more than three sentences) on why your assigned stance is correct.
If the question is ambiguous, address the assumptions/interpretation associated with your assigned stance.
Be concise! Exclude anything irrelavant or unhelpful in terms of supporting the stance!
Argument:
"""
        res = (
            self.chain(
                ChatPromptTemplate.from_messages(
                    [HumanMessagePromptTemplate.from_template(template=prompt)]
                )
            )
            .predict(
                callbacks=self.callback_handlers,
            )
            .strip()
        )
        logging.debug(f"generate_argument ({self.model_type}) {prompt}\n\n{res}")
        argument = res.replace("Argument:", "").strip()
        return argument

    def get_agent_parameters(self) -> Dict[str, str]:
        return {
            item[0]: str(item[1])
            for item in self.__dict__.items()
            if item[0]
            not in [
                "model_wrapper",
                "message_history",
                "memory",
                "prompts",
                "callback_handler",
            ]
        }
