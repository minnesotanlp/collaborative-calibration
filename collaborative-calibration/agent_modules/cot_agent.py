import logging
from typing import Tuple
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from .base_agent import BaseAgent


class CoTAgent(BaseAgent):
    """Class for Chain-of-Thought style agents"""

    system_msg = "You are a helpful AI agent good at multi-hop and arithmetic reasoning."
    abstain_msg = "If you don't think the question can be solved step by step, simply say 'Abstain' and nothing else."

    def retrieve_demonstrations(self, query: str, n_shot: int = 3) -> str:
        """Retrieve relevant memory pieces as few-shot demonstrations."""
        return "\n===\n".join(
            [doc.page_content for doc in self.memory.fetch_memories(query)][:n_shot]
        )

    def self_deliberate_with_pretrained_instance(self, query: str) -> Tuple[float, str]:
        return super().self_deliberate_with_pretrained_instance(query)

    def self_deliberate(self, query: str, n_shot: int = 0) -> str:
        """Self-delibrate candidate answers following CoT prompting."""
        sys_msg = SystemMessage(content=self.system_msg)
        if n_shot == 0:
            # zero-shot CoT
            zs_messages = [
                sys_msg,
                HumanMessagePromptTemplate.from_template(
                    template="Q:{query}\nLet's think step by step. Notice that {abstain_msg})"
                ),
            ]
            intermediate = (
                self.chain(ChatPromptTemplate.from_messages(zs_messages))
                .predict(
                    query=query,
                    abstain_msg=self.abstain_msg,
                    callbacks=self.callback_handlers
                )
                .strip()
            )
            logging.debug(
                f"{self.model_type}, {self.model_wrapper._default_params}, intermediate: {intermediate}"
            )
            if "Abstain" in intermediate:
                res = "Abstain"
            else:
                res = (
                    self.chain(
                        ChatPromptTemplate.from_messages(
                            [
                                HumanMessagePromptTemplate.from_template(
                                    template="Question:{query}\n\{intermediate}"
                                ),
                                HumanMessage(content=self.default_trigger_prompt),
                            ]
                        )
                    )
                    .predict(
                        query=query,
                        intermediate=intermediate,
                        callbacks=self.callback_handlers,
                    )
                    .strip()
                )

        else:
            # few-shot CoT
            demonstrations = self.retrieve_demonstrations(query, n_shot)
            messages = [
                sys_msg,
                AIMessage(content=demonstrations),
                HumanMessagePromptTemplate.from_template(
                    template="Question:{query}\nAnswer the question following the above demonstrations. {trigger_msg}"
                ),
            ]
            res = (
                self.chain(ChatPromptTemplate.from_messages(messages))
                .predict(
                    query=query,
                    trigger_msg=self.default_trigger_prompt,
                    callbacks=self.callback_handlers
                )
                .strip()
            )

        logging.debug(f"{self.model_type}, res: {res}")
        return res
