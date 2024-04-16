import logging
from typing import Any, Tuple, List, Optional
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
 
from load_pretrained import causal_lm_generate
from .base_agent import BaseAgent
from simulation_utils import get_model_specific_chat_template, google_search, wikipedia_search


class SearchAgent(BaseAgent):
    """Class for search-augmented agents with access to Google Search or Wikipedia API
    """

    abstain_msg = "If you don't think the question can be solved from web search, generate 'Abstain' and nothing else."

    def construct_chat_messages(self, prompts: List[str], sys_msg: Optional[str] = None) -> ChatPromptTemplate:
        messages = []
        if sys_msg:
            messages.append(SystemMessagePromptTemplate.from_template(template=sys_msg)) 
        messages.extend([HumanMessagePromptTemplate.from_template(template=msg) for msg in prompts])
        return ChatPromptTemplate.from_messages(messages)


    def fuzzy_search(self, query: str, raw: str) -> str:
        candidates = raw.replace("Could not find exact results. Similar:", "").strip().replace("[", "").replace("]", "").split(",")
        prompt = self.construct_chat_messages([f"Identify the most relevant keyword (a word or a phrase) regrading answering this question: {query}\nKeyword:"])
        keyword = self.chain(prompt).predict(callbacks=self.callback_handlers).strip()
        if keyword not in candidates:
            prompt = self.construct_chat_messages([f"Given this question: {query}\n, identify the most relevant keyword from this list {candidates}\nKeyword:"])
            keyword = self.chain(prompt).predict(callbacks=self.callback_handlers).strip()
        res = wikipedia_search(keyword)
        return res        

    def search(self, query: str, source: str = "google-search", n_entry: int = 1, use_chain: bool = True) -> Tuple[Any, str]:
        if source == "google-search":
            raw_res = google_search(query, n_entry)
        elif source == "wikipedia":
            raw_res = wikipedia_search(query) 
            if "Could not find exact results." in raw_res and use_chain:
                # search in similar Wiki pages
                raw_res = self.fuzzy_search(query, raw_res)
        else:
            raise NotImplementedError
        res = raw_res.replace("{", "{{").replace("}", " }}")
        sys_msg_summary = f"According to the search results (each contains a title and a description), answer this question: {query}"
        sys_msg_summary += self.abstain_msg + "\n" + self.default_trigger_prompt 
        if use_chain:
            prompt = self.construct_chat_messages(prompts=[res], sys_msg=sys_msg_summary)
            prob = 1
            final_ans = self.chain(prompt).predict(callbacks=self.callback_handlers).strip()
        else:
            chat_prompt = get_model_specific_chat_template(self.model_type, f"{res}\n{sys_msg_summary}")
            prob, final_ans = causal_lm_generate(self.pretrained[0], self.pretrained[1], chat_prompt, return_joint_prob=False)
            logging.debug(
                f"{self.model_type}, {type(self.pretrained[1])}, search final_ans: {final_ans}"
            )
        return prob, final_ans

    def self_deliberate_with_pretrained_instance(self, query: str) -> Tuple[float, str]:
        prob, search_res = self.search(query, source="google-search", use_chain=False)
        logging.debug(f"{self.model_type}, search_res: {prob} {search_res}")
        if "Abstain" in search_res:
            prob, res = 0.0, "Abstain"
        else:
            res = search_res  
        return prob, res

    def self_deliberate(self, query: str) -> str:
        _, search_res = self.search(query, source="google-search")
        logging.debug(f"{self.model_type}, search_res: {search_res}")
        if "Abstain" in search_res:
            res = "Abstain"
        else:
            res = search_res  
        return res
