import logging

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from simulation_utils import call_api, get_last_line, google_search, wikipedia_search, extract_conf

from .base_agent import BaseAgent



class SelfAskAgent(BaseAgent):
    """Class for search augmented self-ask style agents
    Adapted from https://github.com/ofirpress/self-ask/blob/main/self-ask_plus_search-engine_demo.ipynb
    """

    intermediate_trigger = "\nIntermediate answer:"
    followup_trigger = "Follow up:"
    termination_trigger = "So the final answer is:"
    final_trigger = (
        "\nGive the final answer (as short as possible, in a few words), and a confidence score (a float from 0 to 1) surrounded in round brackets"
        "follow this format: short answer (confidence)"
    )
    sa_demo = [
        """Question: Who lived longer, Muhammad Ali or Alan Turing?
            Are follow up questions needed here: Yes.
            Follow up: How old was Muhammad Ali when he died?
            Intermediate answer: Muhammad Ali was 74 years old when he died.
            Follow up: How old was Alan Turing when he died?
            Intermediate answer: Alan Turing was 41 years old when he died.
            So the final answer is: Muhammad Ali 

            Question: When was the founder of craigslist born?
            Are follow up questions needed here: Yes.
            Follow up: Who was the founder of craigslist?
            Intermediate answer: Craigslist was founded by Craig Newmark.
            Follow up: When was Craig Newmark born?
            Intermediate answer: Craig Newmark was born on December 6, 1952.
            So the final answer is: December 6, 1952

            Question: """,
        f"""\nAre follow up questions needed here:""",
    ]
    model_stops = None # followup_trigger

    def search(self, query: str, source: str = "google-search", n_entry: int = 3) -> str:
        if source == "google-search":
            raw_res = google_search(query, n_entry)
        elif source == "wikipedia":
            raw_res = wikipedia_search(query) 
        else:
            raise NotImplementedError

        sys_msg_summary = f"According to the following search results (each contains a title and a description), answer the following question: {query}"
        return call_api(prompt=raw_res, sys_msg=sys_msg_summary)


    def extract_question(self, generated: str):
        last_line = get_last_line(generated)
        if "Follow up:" not in last_line:
            logging.info(f"Wrong format: {generated}")
        parts = generated.split(":")
        question = parts[-1].strip()

        if "?" != question[-1]:
            logging.info(f"Wrong format: {generated}")

        return question

    def self_deliberate_with_pretrained_instance(self):
        pass

    def parse_final_output(self, raw: str) -> str:
        parts = raw.split("(")
        if len(parts) != 2:
            logging.debug(f"parts: {parts}")
            parts = raw.split(" ")
        res = f"Answer:{parts[0].replace('Final Answer:', '').replace('Final answer:', '').replace('The final answer is', '').strip()}\n"
        res += f"Confidence:{extract_conf(parts[-1])}"
        return res
    
    def self_deliberate(self, query: str) -> str:
        """Self-delibrate candidate answers following self-ask prompting."""

        cur_prompt = self.sa_demo[0] + query + self.sa_demo[1]

        ret_text = (
            self.chain(
                ChatPromptTemplate.from_messages(
                    [HumanMessagePromptTemplate.from_template(cur_prompt)]
                )
            )
            .predict(
                callbacks=self.callback_handlers
            )
            .strip()
        )
        logging.debug(f"initial ret_text: {ret_text}")
        if "Abstain" in ret_text:
            ret_text = "Abstain"
        else:
            while self.followup_trigger in get_last_line(ret_text):
                cur_prompt += ret_text
                question = self.extract_question(ret_text)
                external_answer = self.search(question, source="google-search")
                logging.debug(f"external_answer: {external_answer}")

                if external_answer is not None:
                    cur_prompt += self.intermediate_trigger + " " + external_answer + "."
                    logging.debug(f"cur_prompt: {cur_prompt}")
                    ret_text = (
                        self.chain(
                            ChatPromptTemplate.from_messages(
                                [HumanMessagePromptTemplate.from_template(cur_prompt)]
                            )
                        )
                        .predict(
                            callbacks=self.callback_handlers
                        )
                        .strip()
                    )

                else:
                    logging.info("Search returns no answer.")
                    cur_prompt += self.intermediate_trigger
                    gpt_answer = call_api(
                        cur_prompt, stop=["\n" + self.followup_trigger, self.final_trigger]
                    )
                    cur_prompt += gpt_answer

            if self.termination_trigger not in ret_text:
                cur_prompt += self.termination_trigger
                logging.debug(f"cur_prompt here: {cur_prompt}")
                ret_text = call_api(cur_prompt, stop=["\n"])

            # parse unified result format
            ret_text = self.parse_final_output(ret_text)

        final = cur_prompt + "\n" + ret_text
        logging.debug(f"{self.model_type}, final: {final}")

        return ret_text