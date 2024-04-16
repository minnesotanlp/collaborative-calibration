import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.chains import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.utils import mock_now


# Prompt for rating memory importance (exicitment score)
EXCITEMENT_SCORE_PROMPT = """
On the scale of 1 to 10, where 1 is completely not engaging or exciting (something that is routine or mundane), \
and 10 is extremely engaging and exciting (something one would memorize vividly, such as an interesting anecdote), \
rate the likely exicitment of the following piece of memory to {name}. Respond with a single integer.
Memory: '{memory_content}'
Rating (integer from 1 to 10):
"""

# Prompt for generating high-level reflection and summary for the buffer memory
REFLECTION_PROMPT = """
Play the role of {name}. Recall that you had a debate with others on a multiple choice question.
Here is the debate history: \n
'''\n{buffer}\n'''
\nGive a brief summary of the arguments of the debate from {name}'s perspective and draw a few insights/observations. \
Pay more attention to your own answers, opinions and arguments across different rounds.
Remember to generate in the position of {name} and use the first person! Pay attention to your exact choices and be factual! \
"""


logger = logging.getLogger(__name__)


class BaseAgentMemory(BaseMemory):
    llm: BaseLanguageModel
    memory_retriever: TimeWeightedVectorStoreRetriever
    reference: str
    token_limit: int = 1000
    reflecting: bool = True
    reflection_threshold: float = 1.0
    forgeting: bool = True
    buffer_length_limit: int = 500
    verbose: bool = True  # for debugging
    num_reflection_entries: int = 4
    importance_weight: float = 0.4
    aggregate_excitement: float = 0.0
    special_focus_scalar: float = 1.5
    reflection_stream: list[Tuple[str, Optional[str]]] = []

    # input keys
    queries_key = "queries"
    most_recent_memories_token_key = "recent_memories_token"
    sensory_memory_key = "sensory_memory"
    buffer_memory_key = "buffer_memory"
    topic_key = "topic"
    # output keys
    relevant_memories_key = "relevant_memories"
    most_recent_memories_key = "most_recent_memories"
    time_key = "timestamp"
    special_focus_key = "special_focus"


    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def get_excitement_score(
        self, sensory_memory: str, buffer_memory: str, special_focus: bool
    ) -> float:
        prompt = PromptTemplate.from_template(EXCITEMENT_SCORE_PROMPT)
        score_raw = (
            self.chain(prompt)
            .run(
                memory_content=f"({sensory_memory}) {buffer_memory}",
                name=self.reference,
            )
            .strip()
        )
        if self.verbose:
            logger.info(f"Excitement score raw: {score_raw}")
        match = re.search(r"^\D*(\d+)", score_raw)
        if not match:
            return 0.0
        scalar = self.special_focus_scalar if special_focus else 1.0
        return max(float(match[1]) * scalar, 10) / 10 * self.importance_weight

    def fetch_memories(self, input: str, now: Optional[datetime] = None) -> List[Document]:
        """Fetch related memories."""
        if now is None:
            return self.memory_retriever.get_relevant_documents(input)
        with mock_now(now):
            return self.memory_retriever.get_relevant_documents(input)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content in content_strs:
                continue
            content_strs.add(mem.page_content)
            created_time = mem.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
            content.append(f"- {created_time}: {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])

    def get_reflection(self, topic: str, time: Optional[datetime] = None) -> str:
        """Get summary (reflection) on recent and important memories"""
        if self.verbose:
            logging.info(f"Reflecting previous memories on {topic} at {time}.")
        prompt = PromptTemplate.from_template(REFLECTION_PROMPT)
        fetched_memories: list[Document] = self._get_memories_until_limit(  # type: ignore
            -1, return_string=False
        )
        buffer = "\n".join([memory.page_content for memory in fetched_memories])
        return self.chain(prompt).run(topic=topic, buffer=buffer, name=self.reference).strip()

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory."""
        logging.debug(f"inputs: {inputs}; outputs: {outputs}")
        sensory_mem = outputs.get(self.sensory_memory_key, None)
        buffer = outputs.get(self.buffer_memory_key)
        topic = outputs.get(self.topic_key)
        time = outputs.get(self.time_key)
        special_focus = outputs.get(self.special_focus_key, None)
        self.add_memory(sensory_mem, buffer, topic, time, special_focus)

    def add_memory(
        self,
        buffer_memory: str,
        sensory_memory: Optional[str] = "",
        topic: Optional[str] = "",
        time: Optional[datetime] = None,
        special_focus: Optional[bool] = False,
    ) -> list[str]:
        """Add relevant memory content to the agent's internal memory
        
        excitement score = f(M(sensory, buffer), special_focus)
        """
        excitement_score = (
            self.get_excitement_score(sensory_memory, buffer_memory, special_focus)
            if self.reflecting
            else 0.0
        )
        self.aggregate_excitement += excitement_score
        document = Document(
            page_content=f"{buffer_memory}",
            metadata={
                "excitement": excitement_score,
                "sensory_memory": sensory_memory,
                "topic": topic,
            },
        )
        result = self.memory_retriever.add_documents([document], current_time=time)
        if (
            sensory_memory
            and self.reflecting
            and self.aggregate_excitement > self.reflection_threshold
        ):
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S") if time else ""
            self.reflection_stream.append((self.get_reflection(topic, time), timestamp_str))
            self.aggregate_excitement = 0.0
        return result

    def _get_memories_until_limit(
        self, consumed_tokens: int, return_string: bool = True
    ) -> Union[str, list[Document]]:
        """Get documents until reaching max number of returned tokens.
        
        If consumed_tokens is -1, get all documents
        """
        result: list[Document] = []
        if consumed_tokens == -1:
            result = self.memory_retriever.memory_stream[::-1]
        else:
            for doc in self.memory_retriever.memory_stream[::-1]:
                if consumed_tokens >= self.token_limit:
                    break
                consumed_tokens += self.llm.get_num_tokens(doc.page_content)
                if consumed_tokens < self.token_limit:
                    result.append(doc)
        return self.format_memories_detail(result) if return_string else result

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        queries = inputs.get(self.queries_key)
        time = inputs.get(self.time_key)
        if queries is not None:
            relevant_memories = [
                mem for query in queries for mem in self.fetch_memories(query, now=time)
            ]
            return {
                self.relevant_memories_key: self.format_memories_detail(relevant_memories),
            }

        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key)
        if most_recent_memories_token is not None:
            return {
                self.most_recent_memories_key: self._get_memories_until_limit(  # type: ignore
                    most_recent_memories_token
                )
            }
        return {}

    def clear(self) -> None:
        """Remove older memory contents (entries outside of context window)."""
        docs = self.memory_retriever.memory_stream
        if self.forgeting and len(docs) > self.buffer_length_limit:
            self.memory_retriever.memory_stream = docs[: self.buffer_length_limit]
