import logging
from typing import Tuple
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, AIMessage
 
from load_pretrained import causal_lm_generate
from .base_agent import BaseAgent
from simulation_utils import get_model_specific_chat_template


class KnowledgeAgent(BaseAgent):
    """Class for knowledge augmented agents for knowledge-intensive or commonsense tasks
    For now, use GENREAD (Yu et.al 2023) in a zero-shot setting (single background document) for simplicity.
    Could also use a similar Generated-Knowledge-Prompting (Liu et.al 2021; yet being used in Si et.al 2023),
        or perhaps Maieutic-prompting (Jung et.al 2022; yet a bit too complex)
    """

    abstain_msg = "If you don't think the question can be solved by generating background context, simply generate 'Abstain' and nothing else."

    def self_deliberate_with_pretrained_instance(self, query: str) -> Tuple[float, str]:
        prompt_intermediate = f"Generate a short paragraph as background document to answer the given question:\n{query}"
        chat_prompt = get_model_specific_chat_template(self.model_type, prompt_intermediate)
        _, doc = causal_lm_generate(self.pretrained[0], self.pretrained[1], chat_prompt, return_joint_prob=False)
        logging.debug(
            f"{self.model_type}, {type(self.pretrained[1])}, doc: {doc}"
        )
        if "Abstain" in doc:
            prob, res = 0.0, "Abstain"
        else:
            prob, res = causal_lm_generate(self.pretrained[0], self.pretrained[1], f"{doc}\n{self.default_trigger_prompt}", return_joint_prob=True)
            logging.debug(f"{res}; model prob: {prob}")
        return prob, res

    def self_deliberate(self, query: str):
        zs_messages = [
            HumanMessagePromptTemplate.from_template(
                f"Generate a background document to answer the given question:\n{query}\n({self.abstain_msg}; The background should be brief, in no more than 100 words)"
            ),
        ]
        doc = (
            self.chain(ChatPromptTemplate.from_messages(zs_messages))
            .predict(
                callbacks=self.callback_handlers
            )
            .strip()
        )
        logging.debug(f"{self.model_type}, doc: {doc}")
        if "Abstain" in doc:
            res = "Abstain"
        else:
            zs_messages.extend([AIMessage(content=doc), HumanMessage(content=self.default_trigger_prompt)])
            res = (
                self.chain(ChatPromptTemplate.from_messages(zs_messages))
                .predict(
                    callbacks=self.callback_handlers
                )
                .strip()
            )
        logging.debug(f"{self.model_type}, res: {res}")
        return res
