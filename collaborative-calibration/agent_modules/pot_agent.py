import logging
from typing import Tuple 
import func_timeout
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage

from .base_agent import BaseAgent
from simulation_utils import extract_conf, get_model_specific_chat_template
from load_pretrained import causal_lm_generate

class POTAgent(BaseAgent):
    """Class for Program-of-Thought style agents (currently only 0-shot)
    Adapted from https://github.com/wenhuchen/Program-of-Thoughts
    """

    system_msg = """
    You are an excellent Python programmer.
    Try to reason the following question by completing a Python function called solver().
    Don't explain or test the code, just generate the solver() function itself. Any line starting with a "#" symbol is a comment.
    """
    abstain_msg = "Do you think the question can be solved by transforming into a Python program? Simply output 'Yes' or 'No'."
    original_template = """
    import math
    import numpy as np

    # Question: {question}
    # Answer this question by implementing a solver() function that returns the answer.
    def solver():
        # Let's write a Python program step by step, and then return the answer
        # Firstly, we need define the following variable:
    """
    trigger_msg = (
        "Given the following question: '{question}' Here is an answer obtained through program execution: '{ans}'"
        "\nGive a float (between 0 to 1) indicating your confidence on how likely the answer is correct"
        "Just a float, no explanations!"
    )
    
    def parse_program(self, raw_program: str) -> str:
        program = raw_program.replace("```python", "").replace("```", "").strip()
        if "mistral" in self.model_type:
            program = program[program.find("def "):]
        body = []
        for line in program.split("\n"):
            body.append(line)
            if "return" in line:
                break            
        body = "\n".join(body)
        if "numpy" not in body:
            body = "import numpy as np\n" + body
        if "math" not in body:
            body = "import math\n" + body
        logging.debug(f"parsed program: {body}")   
        return body 

    def safe_execute(self, code_string: str, keys=None):
        def execute(x):
            try:
                logging.debug("Trying program execution...")
                if "ans = solver()" not in x:
                    x += "\nans = solver()"
                exec(x)                
                locals_ = locals()
                logging.debug(f"code'''\n{x}\n''' locals: {locals_}")
                if keys is None:
                    return locals_.get("ans", None)
                else:
                    return [locals_.get(k, None) for k in keys]
            except Exception as e:
                logging.debug(f"Exception in program execution {str(e)}")
                return None

        try:
            ans = func_timeout.func_timeout(5, execute, args=(code_string,))
        except func_timeout.FunctionTimedOut as e:
            logging.debug(f"Exception in program execution {str(e)}")
            ans = None

        return ans
    
    def self_deliberate_with_pretrained_instance(self, query: str) -> Tuple[float, str]:
        prompt_intermediate = (
            f"Question: {query}\nAnswer the question by implementing a solver() function in Python that returns the answer."
            "Don't explain or test the code, just generate the solver() function itself. Import any library if necessary."
            "Note the solver function shouldn't take any argument. Define any variable inside the function."
        )
        chat_prompt = get_model_specific_chat_template(self.model_type, prompt_intermediate)
        _, program = causal_lm_generate(self.pretrained[0], self.pretrained[1], chat_prompt, return_joint_prob=False)
        logging.debug(
            f"{self.model_type}, {type(self.pretrained[1])}, program: {program}"
        )
        if "Abstain" in program:
            prob, res = 0.0, "Abstain"
        else:
            ans = self.safe_execute(self.parse_program(program))
            prob, res = causal_lm_generate(self.pretrained[0], self.pretrained[1], f"{self.trigger_msg.format(question=query, ans=ans)}", return_joint_prob=True)
            res = f"Answer:{ans}\nConfidence:{res}"

        logging.debug(f"{self.model_type}, res: {res}")
        return prob, res

    def self_deliberate(self, query: str) -> str:
        """Self-delibrate candidate answers following (0-shot) PoT style prompting."""
        pre = self.chain(ChatPromptTemplate.from_messages([HumanMessage(content=f"Question: {query}\n{self.abstain_msg}")])).predict(
            question=query,
            allbacks=self.callback_handlers
        )
        if "no" in pre.lower():
            res = "Answer:\nConfidence:-1"
        else:
            messages = [
                SystemMessage(content=self.system_msg),
                HumanMessagePromptTemplate.from_template(
                    template=f"Let's write a Python program step by step:\n{self.original_template}"
                ),
            ]
            program = self.chain(ChatPromptTemplate.from_messages(messages)).predict(
                question=query,
                allbacks=self.callback_handlers
            )
            logging.debug(f"{self.model_type}, program: {program}")
            ans = self.safe_execute(self.parse_program(program))
            if ans:
                conf = self.chain(ChatPromptTemplate.from_messages([HumanMessage(content=self.trigger_msg.format(question=query, ans=ans))])).predict(
                    callbacks=self.callback_handlers
                )
                res = f"Answer:{ans}\nConfidence:{extract_conf(conf)}"
            else:
                res = "Abstain"

        buffer = f"\nQuestion: {query}\n{res}"
        logging.debug(f"{self.model_type}, buffer: {buffer}")
        return res