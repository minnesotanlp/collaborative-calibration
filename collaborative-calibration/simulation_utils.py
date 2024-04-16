from ast import literal_eval
import logging
import os
import itertools
import re
import random
import time
import requests
import backoff
from requests.exceptions import RequestException
from typing import Any, Dict, List, Optional, Tuple, Union


import wikipedia
from web_assets import DOMAINS, USER_AGENTS
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup

import numpy as np
import openai
import pandas as pd
import torch
import cohere

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TEST_API_MODELS = {
    "openai-chat": [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-4",
        "gpt-4-1106-preview",
        # "gpt-4-turbo",
    ],
    "openai-completion": ["gpt-3.5-turbo-instruct", "text-davinci-003"],
    "cohere": ["cohere-generate"],
}

TEST_HF_MODELS = {
    "meta-llama": ["Llama-2-7b", "Llama-2-13b", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"],
    "lmsys": ["vicuna-13b-v1.3"],
    "mistralai": ["Mistral-7B-Instruct-v0.1"]
}

def extract_ans(res: str) -> str:
    if res.find("A:") == -1 and res.find("Answer:") == -1:
        logging.debug(f"wrong output format: {res}")
        return "Abstain"
    if "<answer>" in res or "I'm sorry" in res:
        logging.debug(f"Abstain from answering: {res}")
        return "Abstain"
    return (
        res.split("\n")[0]
        .split("Confidence")[0]
        .split("confidence")[0]
        .replace("A:", "")
        .replace("Answer:", "")
        .replace("{", "{{")
        .replace("}", "}}")
        .strip()
    )


def extract_conf(res: str) -> float:
    res_parsed = (
        res.split("\n")[-1]
        .replace("Confidence:", "")
        .replace("Confidence score:", "")
        .replace("Confidence Score:", "")
        .replace("confidence score:", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )
    try:
        conf = float(res_parsed)
    except ValueError:
        logging.debug(f"res {res} res_parsed: {res_parsed}")
        conf = 0.0
    return conf

def extract_conf_metrics(res: str) -> Tuple[float, float, float]:
    res_parsed = {"ambiguity": None, "complexity": None, "ability": None}
    for line in res.split("\n"):
        text = line.strip().lower()
        for k in res_parsed.keys():
            if text.startswith(f"{k}:"):              
                try:
                    text_val = re.sub("[\(\[].*?[\)\]]", "", text.split(f"{k}:")[-1])
                    metric = float(text_val.strip())
                except ValueError:
                    logging.debug(f"metric_name {k} text: {text}")
                    metric = None
                res_parsed[k] = metric
    logging.debug(f"conf_metrics parsed {res_parsed}")
    return res_parsed

def get_model_specific_chat_template(model_type: str, user_msg: str) -> str:
    if "meta-llama/Llama-2" in model_type:
        return (
            "<s>[INST] <<SYS>>\nYou are a helpful AI agent. Be factual and truthful!\n<</SYS>>"
            f"[INST] {user_msg} [/INST]"
    )
    elif model_type == "mistralai/Mistral-7B-Instruct-v0.1":
        return (
            f"<s>[INST] You are a helpful AI agent. [/INST] Hi! I will be helpful, factual and truthful.</s> [INST] {user_msg} [/INST]"
        )
    elif model_type == "lmsys/vicuna-13b-v1.3":
        return (
            "You are a helpful AI agent.\n\n"
            f"USER: Be helpful, factual and truthful!\nASSISTANT: OK!</s>\nUSER: {user_msg}\nASSISTANT:"
        )
    else:
        raise NotImplementedError


def eval_ans(
    query: str,
    ans: str,
    reference: Union[str, List[str]],
    method: str = "gpt_cls",
    nli_classifier: Optional[Any] = None,
    nli_tokenizer: Optional[Any] = None,
    openai_api_key: Optional[str] = None,
) -> bool:
    """Evaluate whether the provided answer is semantically equivalent to the gold reference
    For now use a GPT judge. Could also use exact-match or a separately trained entailment classifier (require parsing for multiple references)
    """
    if isinstance(reference, str):
        reference = [reference]
    if method == "em":
        return any([ans == ref for ref in reference])
    elif method == "gpt_cls":
        for ref in reference:
            if ans == ref:
                return True
            judgement = call_api(
                prompt=(
                    f"Are the following two answers to my question Q semantically equivalent? \n\nQ:'{query}',"
                    f"\nA1:'{ans.lower()}'\nA2:'{ref.lower()}'"
                    f"\n\nSimply output 'Yes' or 'No'"
                ),
                api_key=openai_api_key
            )
            if "Yes" in judgement:
                return True
        return False
    elif method == "nli_cls":
        for ref in reference:
            e1 = check_entailment(ans, ref, nli_tokenizer, nli_classifier)
            e2 = check_entailment(ref, ans, nli_tokenizer, nli_classifier)
            if e1 == e2 == 1:
                return True
        return False
    else:
        raise NotImplementedError


def summarize_feedback(feedback: List[str], rating_scales: Optional[List[str]] = ["bad", "modest", "good", "excellent"]) -> str:
    summary_sys_msg = "Be factual and concise in your summarization."
    prompt = (
        f"Summarize by combining the feedback from several individuals: {feedback}"
        f" Note the rating scales are {rating_scales}. You should aggregate the ratings from both sides."
        " Also, highlight any unfactual premise mentioned.\n"
        "Summary:"
    )
    summary_all = call_api(prompt, summary_sys_msg, model="gpt-3.5-turbo-1106")
    logging.debug(f"summary_all: {summary_all}")
    return summary_all


def check_entailment(premise: str, hypothesis: str, tokenizer: Any, classifier: Any) -> int:
    """Check for entailment/neutral/contradiction using a NLI classifier (class order ["entailment", "neutral", "contradiction"])
    return 1 if "entailment", 0 if "neutral", and -1 if "contradiction"
    """
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = classifier(input["input_ids"])
    prediction = np.argmax(torch.softmax(output["logits"][0], -1).tolist())
    logging.info(
        f"entailment prediction: {prediction}, and {torch.softmax(output['logits'][0], -1).tolist()}"
    )
    return 1 - prediction


def give_verbal_feedback(stance: str, argument: str, consistency_level: int) -> str:
    prompt = f"For the given stance '{stance}', someone argued:\n'{argument}'\n"
    if consistency_level == -1:
        prompt += (
            "It is found that there is major inconsistency between the stance and the argument.\n"
        )
    prompt += "Evaluate the argument in terms of logical consistency, factuality, clarity, and conciseness."
    prompt += "Give a rating (a float from 0 to 1) considering the above aspects, and give reasons for your rating."
    return call_api(prompt)


def generate_premises_and_ratings(
    argument: str, stance: str, scale: List[str], notice: str, verify: Optional[bool] = False
) -> Tuple[List[Optional[str]], Dict[str, str]]:
    verify_text = (
        "List premises/assumptions from the above argument that help determine the correctness of the stance"
        " Each premise should be as simple as possible (single-hop), relevant to the stance, and independent"
        " (avoid pronoun as the subject, use named entity instead), surrounded in square brackets, e.g. [Premise:<premise>]"
    ) if verify else ""
    prompt = (
        f"Here is an argument '{argument}' for the stance '{stance}'\n{notice}\n"
        f"{verify_text}"
        "Evaluate how good the argument is regarding logical consistency, clarity, and conciseness. "
        f"For each of the three aspects, choose one of {scale} as your rating. Do NOT provide any reasoning."
        "Follow this format: [Consistency:<rating>, Clarity:<rating>, Conciseness:<rating>]"
    )
    raw_res = call_api(prompt).split("\n")
    premises_parsed, ratings = [], {
        "Consistency": "modest",
        "Clarity": "modest",
        "Conciseness": "modest",
    }
    for row in raw_res:
        if "Premise" in row:
            premises_parsed.append(
                row.replace("Premise", "")
                .replace(":", "")
                .replace("[", "")
                .replace("]", "")
                .strip()
            )
        elif "Consistency:" in row:
            parts = row.strip().split(",")
            if len(parts) != 3:
                logging.debug(f"ratings wrong format {row}")
            ratings_parsed = [part[part.find(":") + 1 :].replace(".", "").replace("]", "").strip() for part in parts]
            ratings.update(zip(ratings, ratings_parsed))
    logging.debug(f"raw_res: {raw_res}")
    return premises_parsed, ratings


def verify_premises(
    premises: List[str], nli_tokenizer: Any, nli_classifier: Any, verify_with_search: bool = True,
) -> Tuple[float, str]:
    """Verify premises by self-ask"""
    verification_results = [0] * 3  # "True", "Unknown", "False"
    cases = ""
    for premise in premises:
        q_rewrite = call_api(
            prompt=f"Rewrite the premise '''{premise}''' to a question starting with an interrogative pronoun:"
        ).strip()        
        if verify_with_search:
            sys_msg_summary = f"According to the following search results, answer the following question: {q_rewrite}"
            ans_self_ask = call_api(prompt=google_search(q_rewrite, n_entry=1), sys_msg=sys_msg_summary)
        else:
            ans_self_ask = call_api(
                prompt=f"Answer the question: {q_rewrite}. If you are unsure about the correct answer, simply output 'Unknown'"
            )
        if "Unknown" in ans_self_ask:
            verification_results[1] += 1
        else:
            nli_cls = check_entailment(ans_self_ask, premise, nli_tokenizer, nli_classifier)
            if nli_cls == -1:
                check = call_api(
                    prompt=f"Is the statement {premise} True or False? Briefly explain why."
                ).strip()
                if "False" in check:
                    cases += f"For {premise}: {check}\n"
                else:
                    nli_cls = 0
            verification_results[1 - nli_cls] += 1

            logging.debug(f"verify_premise res: {premise}; {q_rewrite}; {ans_self_ask}; {nli_cls}")

    # factuality_score in [0,1]
    factuality_score = (
        1 * verification_results[0] + 0.5 * verification_results[1] + 0 * verification_results[2]
    ) / sum(verification_results)

    return factuality_score, cases

def generate_feedback(
        query: str, 
        answer: str, 
        argument: str,
        nli_tokenizer: Any,
        nli_classifier: Any,
        rating_scales: Optional[List[str]] = ["bad", "modest", "good", "excellent"],
        theta: Optional[float] = 0.6,
        rater_choice: Optional[str] = None,
        verify: Optional[bool] = False,
) -> Tuple[float, str]:
    stance = f"The answer to the question '{query}' is '{answer}'"
    notice = f"Note in earlier debate, you were {rater_choice} the answer corresponding to this argument." if rater_choice in ["supporting", "opposing"] else ""
    ans_consistency = check_entailment(argument, stance, nli_tokenizer, nli_classifier)
    premises, ratings = generate_premises_and_ratings(argument, stance, rating_scales, notice, verify)
    if premises:
        factuality_score, unfactual_premises = verify_premises(
            premises, nli_tokenizer, nli_classifier
        )
    else:
        factuality_score = 1
        unfactual_premises = "No unfactual premise."
    if ans_consistency != 1:
        # neutral (0, downgrade 1 level) or contradictory (-1, downgrade 2 levels)
        adjusted_consistency_level = np.max(
            [0, rating_scales.index(ratings["Consistency"].lower()) + ans_consistency - 1]
        )
        extra = (
            "(The answer cannot be deduced from the arguments)" if ans_consistency == -1 else ""
        )
        ratings["Consistency"] = f"{rating_scales[adjusted_consistency_level]}{extra}"
        
    # rely on model judgement entirely in one call (including reasons for the rating)
    logging.debug(f"ratings values {list(ratings.values())}")
    ratings_numerical = [
        rating_scales.index(grade.split("(")[0].lower()) for grade in ratings.values()
    ]
    soundness_score = float(
        theta * factuality_score
        + (1 - theta) * np.mean(ratings_numerical) / (len(rating_scales) - 1)
    )

    feedback = f"Logic consistency: {ratings['Consistency']}\nClarity: {ratings['Clarity']}\nConciseness: {ratings['Conciseness']}\n{unfactual_premises}\n"
    return soundness_score, feedback
    


def get_collective_feedback(
    query: str,
    arguments: Dict[str, str],
    nli_tokenizer: Any,
    nli_classifier: Any, 
    n_raters: Optional[int] = 1,
    verify: Optional[bool] = False,
) -> List[Tuple[str, str, float, str]]:
    """Rate and rank the arguments based on logical consistency, factuality, clarity, and conciseness."""
    ranking = []  # <answer, argument, soundness_score, verbal_feedback> 
    for answer, argument in arguments.items():
        soundness_scores, feedback_all = [], []
        for _ in range(n_raters):
            score_supporting, feedback_supporting = generate_feedback(query, answer, argument, nli_tokenizer, nli_classifier, rater_choice="supporting")
            score_opposing, feedback_opposing = generate_feedback(query, answer, argument, nli_tokenizer, nli_classifier, rater_choice="opposing", verify=verify)
            soundness_scores.extend([score_supporting, score_opposing])
            feedback_all.extend([feedback_supporting, feedback_opposing])

        ranking.append(tuple(
            [
                answer,
                argument,
                np.mean(soundness_scores),
                summarize_feedback(feedback_all),
            ]
        ))      

    # ranking based on soundness_score
    ranking = sorted(ranking, key=lambda t: t[2], reverse=True)
    return ranking


def sample_input_data(
    dataset_name: str,
    n_test_sample: int = 100,
    n_dev_sample: int = 15,
) -> pd.DataFrame:
    converters = (
        {"reference_answers": literal_eval}
        if dataset_name in ["triviaqa_dev", "truthfulqa", "ambigqa"]
        else None
    )
    df = pd.read_csv(
        f"data/benchmarks/preprocessed/{dataset_name}.csv",
        converters=converters,
    )
    if dataset_name == "ambigqa":
        df["reference_answers"] = [list(itertools.chain(*reference_answers)) for reference_answers in df["reference_answers"].to_list()]
    all = df.sample(n_test_sample + n_dev_sample)
    return all.iloc[:n_test_sample], all.iloc[n_test_sample:]


def split(a: List[Any], n: int):
    """Split list a to n chunks
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1.0,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 16,
    errors: tuple = (openai.OpenAIError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:  # noqa: F841
                logging.debug(f"{e}\nRetrying...")
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@retry_with_exponential_backoff
def call_api(
    prompt: str,
    sys_msg: Optional[str] = None,
    stop: Optional[List[str]] = [],
    model: Optional[str] = "gpt-3.5-turbo-1106",
    api_key: Optional[str] = None,
) -> str:
    """Perform a single api call with specified model and prompt."""
    openai.api_key = os.getenv("OPENAI_API_KEY") or api_key
    if model in TEST_API_MODELS["openai-chat"]:
        messages = [{"role": "system", "content": sys_msg}] if sys_msg else []
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=1024,
            stop=stop,
            request_timeout=900,
        )
        msg = response["choices"][0]["message"]
        assert msg["role"] == "assistant", "Incorrect role returned."
        ans = msg["content"]
    elif model in TEST_API_MODELS["openai-completion"]:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=1.0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        ans = response["choices"][0]["text"]
    elif model in TEST_API_MODELS["cohere"]:
        response = cohere.Client.generate(prompt=prompt, max_tokens=1024, stop_sequences=stop)
        ans = response.text.strip()
    else:
        raise NotImplementedError
    return ans

def backoff_handler(details):
    logging.debug(f"backoff_hdlr: {details}")
    logging.debug("Sleeping...")
    time.sleep(30 * 2 ** details["tries"])

def giveup_handler(details):
    logging.debug(f"Max retries exceeded. {details} Break.")
    return None

### Google Search, adapted from https://github.com/Nv7-GitHub/googlesearch ###
def _req(term, results, lang, start, proxies, timeout):
    agent = np.random.choice(USER_AGENTS)
    domain = np.random.choice(DOMAINS)
    url = "https://{domain}/search?q={q}&hl={hl}&num={n}&start={start}".format(
        q = quote_plus(term),
        domain = domain,
        hl = lang,
        n = results + 2,
        start= start,
    )
    start_time = time.time()
    resp = requests.get(
        url=url,
        headers={
            "user-agent": agent
        },
        proxies=proxies,
        timeout=timeout,
        # allow_redirects=False,
    )
    resp.raise_for_status()     
    
    finish_time = time.time()
    logging.debug(f"Finish searching in {finish_time - start_time}s; status {resp.status_code}")
    time.sleep(np.random.rand()) 
    return resp


class SearchResult:
    def __init__(self, url, title, description):
        self.url = url
        self.title = title
        self.description = description

    def __repr__(self):
        return f"SearchResult(url={self.url}, title={self.title}, description={self.description})"

@backoff.on_exception(
    backoff.expo,
    RequestException,
    max_tries=3,
    raise_on_giveup=False,
    on_backoff=backoff_handler,
    on_giveup=giveup_handler,
)
def search_step(term, num_results=1, lang="en", proxy=None, sleep_interval=0, timeout=5):
    """Search the Google search engine"""

    escaped_term = term.replace(" ", "+")

    # Proxy
    proxies = None
    if proxy:
        if proxy.startswith("https"):
            proxies = {"https": proxy}
        else:
            proxies = {"http": proxy}

    # Fetch
    start = 0
    search_results = []
    while start < num_results:
        # Send request
        resp = _req(escaped_term, num_results - start,
                    lang, start, proxies, timeout)
        if resp:            
            # Parse
            soup = BeautifulSoup(resp.text, "html.parser")
            result_block = soup.find_all("div", attrs={"class": "g"})
            for result in result_block:
                # Find link, title, description
                link = result.find("a", href=True)
                title = result.find("h3")
                description_box = result.find(
                    "div", {"style": "-webkit-line-clamp:2"})
                if description_box:
                    description = description_box.text
                    if link and title and description:
                        start += 1
                        search_results.append(SearchResult(link["href"], title.text, description))
                else:
                    raise RequestException(f"Response in wrong format. Status: {resp.status_code}")
        else:
            raise RequestException(f"Empty response. Status: {resp.status_code}")
        time.sleep(sleep_interval)
    return search_results


def google_search(query: str, n_entry: int = 1) -> str: 
    result = "Search results:\n" 
    res_search_step = search_step(query, num_results=n_entry, sleep_interval=5)
    if res_search_step:
        for i, res in enumerate(res_search_step):
            if i >= n_entry:
                break
            result += f"{i}. title: \"{res.title}\" description: \"{res.description}\"\n\n"

    return result

def wikipedia_search(query: str) -> str:
    '''https://github.com/Gentopia-AI/Gentopia/blob/main/gentopia/tools/wikipedia.py'''
    result = "Search result:\n"
    try:
        title = wikipedia.page(query).title
        result += f"title: {title}, description: {wikipedia.summary(title)}"
    except wikipedia.PageError:
        result = f"Could not find exact results. Similar: {wikipedia.search(query)}"
    except wikipedia.DisambiguationError:
        result = f"Could not find exact results. Similar: {wikipedia.search(query)}"
    return result

def get_last_line(generated: str) -> str:
    if "\n" not in generated:
        return generated
    else:
        return generated.split("\n")[-1]


def softmax(l: List[float]) -> List[float]:
    return np.exp(l) / np.sum(np.exp(l), axis=0)
