"""Self-deliberation Multi-agent Debate Simulator"""
import argparse
import logging
import os
import itertools
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple, Optional, Union

import numpy as np
import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from agent_modules.base_agent import BaseAgent
from agent_modules.cot_agent import CoTAgent
from agent_modules.pot_agent import POTAgent
from agent_modules.search_agent import SearchAgent
from agent_modules.knowledge_agent import KnowledgeAgent

from load_pretrained import load_causal_lm, load_entailment_classifier

from simulation_utils import (
    TEST_API_MODELS,
    TEST_HF_MODELS,
    sample_input_data,
    extract_ans,
    extract_conf,
    extract_conf_metrics,
    eval_ans,
    softmax,
    get_collective_feedback,
)

MEMORY_PATH = "data/memory/"
PROMPTING_STRATEGY_MAPPING = {
    "cot": CoTAgent,
    "pot": POTAgent,
    # "self-ask": SelfAskAgent,
    "search": SearchAgent,
    "knowledge": KnowledgeAgent,
    # "maieutic": MaieuticAgent,
}
MODEL_ENSEMBLE_CHOICES = tuple(
    [
        # "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        # "gpt-4-1106-preview",
        # "gpt-4-turbo",
        "cohere-generate",
        # "meta-llama/Llama-2-13b",
        # "meta-llama/Llama-2-13b-hf",
        # "meta-llama/Llama-2-13b-chat-hf",
        # "lmsys/vicuna-13b-v1.3",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ]
)


def populate_expert_agents(
    selection: Dict[str, int], model_ensemble: bool = False, default_model: str = "gpt-3.5-turbo-1106", mix_temperature: bool = False, use_vllm: bool = False
) -> Dict[str, Any]:
    pool = {k: [] for k in selection.keys()}
    pretrained_instances = {}
    for choice in set(MODEL_ENSEMBLE_CHOICES):
        if model_ensemble and choice.split("/")[-1] in itertools.chain(*TEST_HF_MODELS.values()):
            # load HF instances
            pretrained_instances.update({choice: load_causal_lm(choice, access_token=os.getenv("HF_TOKEN"), use_vllm=use_vllm)})

    for k, count in selection.items():
        for i in range(count):
            temperature = np.random.rand() + 0.5 if mix_temperature else 1
            if k == "gen":
                # if no expertise, must have model ensemble
                for model in MODEL_ENSEMBLE_CHOICES:
                    pool[k].append(
                        BaseAgent(id=f"gen-agent_{i+1}_{model}", model_type=model, pretrained=pretrained_instances.get(model), model_temperature=temperature)
                    )
            elif model_ensemble and k != "self-ask":
                for model in MODEL_ENSEMBLE_CHOICES:
                    pool[k].append(
                        PROMPTING_STRATEGY_MAPPING[k](
                            id=f"{k}-agent_{i+1}_{model}", model_type=model, pretrained=pretrained_instances.get(model), model_temperature=temperature
                        )
                    )
            else:
                pool[k].append(
                    PROMPTING_STRATEGY_MAPPING[k](
                        id=f"{k}-agent_{i+1}_{default_model}", model_type=default_model, model_temperature=temperature
                    )
                )
    logging.debug(f"agents populated: {pool}")
    return pool


def populate_general_agents(size: int) -> List[BaseAgent]:
    return [BaseAgent(id=f"general_{i+1}") for i in range(size)]


def allocate_agent_slots(
    sampled_df: pd.DataFrame, n_slots: int = 12, tau: float = 0.2, use_vllm: bool = False
) -> Dict[str, int]:
    """Allocate n slots of expert agents with different specializations (select the best composition of prompting strategies)
    Current approach:
        Initialize one agent per specialization. On m sampled dev questions, get prediction and confidence.
        Rank agents based on average confidence score adjusted on correctness 
            i=1,...,|skills|; j=1,...,|samples|
            c'_ij =  I(a_ij is "Abstain") * ( 2 * I(a_ij is correct) - 1) * c_ij
        Allocate slots according to agent-wise mean confidence:
            Filter out those with mean confidence below some threshold tau (typically > 0)
            Allocate n slots (roughly) proportional to softmax(C_i)
    """
    # get adjusted confidence
    initialization = {"cot": 1, "pot": 0, "search": 0, "knowledge": 1}
    initial_agents = populate_expert_agents(initialization, use_vllm=use_vllm)
    adjusted_confidence_all = {k: [] for k in initialization.keys()}
    sampled_questions = sampled_df["question"].values.tolist()
    reference_answers = sampled_df["reference_answers"].values.tolist()
    for j, question in enumerate(sampled_questions):
        for key, agent_group in initial_agents.items():
            for agent in agent_group:
                res = agent.self_deliberate(query=question)
                if "Abstain" in res:
                    adjusted_confidence_all[key].append(0.0)
                else:
                    correctness = eval_ans(question, extract_ans(res), reference_answers[j])
                    adjusted_confidence_all[key].append((2 * correctness - 1) * extract_conf(res))
                    if key == "pot":
                        logging.debug(f"pot pair: ({extract_ans(res)}, {reference_answers[j]})->{correctness}")

    logging.debug(f"adjusted_confidence_all: {adjusted_confidence_all}")
    adjusted_confidence_mean = {
        k: float(np.mean(adjusted_confidence_all[k])) for k in initialization.keys() if adjusted_confidence_all[k]
    }
    logging.debug(f"adjusted_confidence_mean: {adjusted_confidence_mean}")

    # agent allocation
    final_allocation = {}
    confidence_filtered = {k: c for k, c in adjusted_confidence_mean.items() if c >= tau}
    logging.debug(f"confidence_filtered: {confidence_filtered} from {adjusted_confidence_mean.items()}")
    if len(confidence_filtered):
        portions = softmax(list(confidence_filtered.values()))
        confidence_softmax = {_k: portions[i] for (i, _k) in enumerate(confidence_filtered.keys())} 
        confidence_sorted = dict(sorted(confidence_softmax.items(), key=lambda item: item[1], reverse=True))
        logging.info(f"{portions}, sorted: {confidence_sorted}")
        sorted_keys = list(confidence_sorted.keys())
        # first allocate proportional to floor(softmax(C_i)), then add remaining slots (if any) to the top-ranked agent
        # edge case: n_slots = 2, then for diversity, initialize the top-2 agents (if over one agent type selected)
        if n_slots == 2 and len(sorted_keys) > 1:
            final_allocation[sorted_keys[0]] = 1
            final_allocation[sorted_keys[1]] = 1
        else:
            for i, k in enumerate(sorted_keys):
                final_allocation.update({k: int(np.floor(portions[i] * n_slots))})
            diff = n_slots - sum(final_allocation.values())
            if diff > 0:
                final_allocation[sorted_keys[0]] += diff

        if sum(final_allocation.values()) != n_slots:
            logging.debug(
                f"slots unmatched: {adjusted_confidence_mean}\n{confidence_filtered}\n{final_allocation}"
            )
            raise ValueError
    else:
        logging.info("All agents produced confidence below threshold. Check input task difficulty or initialization.")
        # in this case, allocate all slots to the most confident agent type
        top_key = max(adjusted_confidence_mean, key=adjusted_confidence_mean.get)
        final_allocation[top_key] = n_slots

    logging.info(f"Final allocation of expert agents (for each model): {final_allocation}")
    return final_allocation


def construct_stances(
    votes: List[Tuple[str, str, str, float, Union[float, None], Dict[str, float]]],
    query: str,
    nli_classifier: Optional[Any],
    nli_tokenizer: Optional[Any],
    filter_abstain: Optional[bool] = True,
    conf_rationales: Optional[List[str]] = None,
) -> List[List[Any]]:
    # dict{ans_class: [mean_verb_confidence, [seq_prob], count, confidence_rationale]}
    classes = {}
    if filter_abstain:
        votes = [vote for vote in votes if "abstain" not in vote[2].lower()]
    for i, (id, model, ans, verb_conf, seq_prob, *_) in enumerate(votes):
        logging.debug(f"construct_stances: {i, id, model, ans, verb_conf, seq_prob}")
        conf_rationale = conf_rationales[i] if conf_rationales else None  # for now, only include in stage 2
        if i == 0:
            classes.update({ans: [verb_conf, [seq_prob], 1, conf_rationale]})
        else:
            merged_answer_class = None
            for unique_class in classes.keys():
                if eval_ans(
                    query,
                    ans,
                    unique_class,
                    method="gpt_cls",  # "nli_cls"
                    nli_classifier=nli_classifier,
                    nli_tokenizer=nli_tokenizer,
                ):
                    # not a new class, merge with the equivalent answer class
                    merged_answer_class = unique_class
                    prev_verb_conf, prev_seq_prob, prev_count, prev_rationale = classes[merged_answer_class]
                    prev_seq_prob.append(seq_prob)
                    classes[merged_answer_class] = [
                        (prev_verb_conf * prev_count + verb_conf) / (prev_count + 1),
                        prev_seq_prob,
                        prev_count + 1,
                        prev_rationale,
                    ]
                    break
            if not merged_answer_class:
                # a new class, add the answer to the answer set
                classes.update({ans: [verb_conf, [seq_prob], 1, conf_rationale]})

    # (unique_ans, mean_verb_confidence, mean_seq_prob, count, confidence_rationale)
    stances = []
    for ans_class, (verb_conf, seq_probs, count, rationale) in classes.items():
        seq_probs_filtered = [seq_prob for seq_prob in seq_probs if seq_prob]
        seq_probs = np.mean(seq_probs_filtered) if seq_probs_filtered else 0
        logging.debug(f"seq_probs_filtered: {seq_probs_filtered}, seq_probs: {seq_probs}")
        stances.append([ans_class, float(verb_conf), float(seq_probs), int(count), rationale])
    logging.debug(f"final ans_set: {stances}")
    return stances


def stance_generation(
    record: pd.Series, agents_mapping: Dict[str, Any], nli_tokenizer: Any, nli_classifier: Any,
) -> Tuple[List[Any], List[Any]]:
    """Stage 1: the selected expert agents vote independently, output a set of semantically unique answers and corresponding confidence/count"""
    votes = []
    for specialization, grouped_agents in agents_mapping.items():
        for agent in grouped_agents:
            if agent.model_type.split("/")[-1] in itertools.chain(*TEST_HF_MODELS.values()) and specialization != "self-ask":
                prob, res = agent.self_deliberate_with_pretrained_instance(record["question"])
            else:
                prob = None
                res = agent.self_deliberate(record["question"])
            vote = tuple((agent.id, agent.model_type, extract_ans(res), extract_conf(res), prob, extract_conf_metrics(res)))
            votes.append(vote)
    logging.info(f"votes: {votes}")
    stances = construct_stances(votes, record["question"], nli_classifier, nli_tokenizer)
    return votes, stances


def revote(
    question: str, mappings: List[Tuple[BaseAgent, float, str, str]]
) -> List[Tuple[str, str, str, float, Union[float, None], str]]:
    votings_all = []
    for mapping in mappings:
        agent, initial_conf, original_observations, new_observations = mapping
        prompt = f"Given the question: '{question}', \n{original_observations}\nHere are some new observations:\n{new_observations}"
        prompt += "Give your final answer (as short as possible). "
        prompt += "Considering your original belief, group consensus and new observations, and weighing arguments from multiple sides (including your own), "
        prompt += "give rationales for whether you would adjust your original confidence score.\nFollow this format:\n"
        prompt += "Answer:\nRationales:"
        revote_intermediate = agent.chain(ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(prompt)])).predict().strip()

        logging.debug(f"revote_intermediate: {revote_intermediate}")
        rationale = revote_intermediate.split("Rationales:")[-1].strip()
        prompt_trigger = f"Recall your orignal confidence for your answer was {initial_conf}. "
        prompt_trigger += f"Given the rationale:\n'''{rationale}'''\nprovide your final confidence score (a float from 0 to 1). Follow this format:\nConfidence:"
        revote_conf = agent.chain(ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(prompt_trigger)])).predict().strip()

        try:
            conf_parsed = float(
                revote_conf.split("Rationales:")[0].split("Confidence:")[-1].split("\n")[0].strip()
            )
        except ValueError:
            logging.debug(f"conf_parsed: {revote_conf}")
            conf_parsed = 1.0
        
        votings_all.append(tuple((mapping[0].id, mapping[0].model_type, extract_ans(revote_intermediate), conf_parsed, None, rationale))) 
    logging.debug(f"votings_all {votings_all}")
    return votings_all


def deliberate_with_feedback(
    question: str,
    agents: List[BaseAgent],
    nli_tokenizer: Any,
    nli_classifier: Any,
    stance_list: List[Tuple[str, float, float, int, Any]],
    group_pruning: Optional[bool] = False,
    long_form: bool = True,
    self_popularity: bool = False,
    verify: bool = False,
) -> Tuple[Dict[str, str], List[Any], Set[Tuple[str, float, float, int, str]]]:
    """Stage 2: m general agents, each with an assigned stance and corresponding confidence (verb/logit-based). 
    Group deliberation process:

    agents: the list of general agents
    stance_list: [<unique_answer, verb_confidence, model_prob, count, conf_rationale>], sorted by count ascendingly
    """
    agents_observations_mapping = []  # [<agent, initial_conf, original observations, new observations/feedback>]
    class_count = [stance_stats[3] for stance_stats in stance_list]
    # generate one argument for each class
    arguments = {t[0]: None for t in stance_list}  # stance: argument
    if len(stance_list) == 1:
        # if reaching consensus in stage 1, assign only 3 general agents, and no feedback needed
        unique_answer, initial_verb_conf, initial_seq_prob, *_ = stance_list[0]
        initial_conf = max(initial_verb_conf, initial_seq_prob)
        original_observations = (
            f"Your original answer is '{unique_answer}', with a confidence of {initial_conf:.2f}"
        )
        new_observations = "Through deliberation, all other people have agreed with your answer, reaching a consensus."
        agents_observations_mapping = [
            tuple((agents[i], initial_conf, original_observations, new_observations)) for i in range(3)
        ]
    else:
        # m general agents, generate arguments and get moderator feedback
        if group_pruning:
            # m = len(agents) < count of independent votes from stage 1 (number of specialized agents that didn't abstain)
            class_freq = class_count / np.sum(class_count)
            logging.debug(f"class_freq: {class_freq}")
            assignment_quantities = np.floor(class_freq * len(agents))
            assignment_quantities = [np.max([quantity, 1]) for quantity in assignment_quantities]
            assignment_quantities[-1] = len(agents) - np.sum(assignment_quantities[:-1])
            assignment_quantities = np.cumsum(assignment_quantities)
        else:
            # exact same assignment as the independent votes from stage 1
            assignment_quantities = np.cumsum(class_count)
        logging.debug(f"assignment_quantities {assignment_quantities}")
        m = assignment_quantities[-1]
        logging.debug(f"agent count: {agents}, stance_list {stance_list} m {m}")

        curr_stance_index = 0
        for index, agent in enumerate(agents):
            # stance_list already sorted
            if index == assignment_quantities[curr_stance_index]:
                curr_stance_index += 1
            if curr_stance_index == len(assignment_quantities):
                break
            assigned_ans, initial_verb_conf, initial_seq_prob, count, _ = stance_list[curr_stance_index]  
            initial_conf = max(initial_verb_conf, initial_seq_prob)         
            agents_observations_mapping.append([agent, assigned_ans, initial_conf, count])
            if long_form:
                arguments[assigned_ans] = agent.generate_self_evaluation(question, assigned_ans)
            else:
                if not arguments.get(assigned_ans):
                    arguments[assigned_ans] = agent.generate_argument(question, assigned_ans)
        logging.info(f"deliberator arguments: {arguments}")

        ranking = get_collective_feedback(question, arguments, nli_tokenizer, nli_classifier, verify)
        # ranking: [<ans, argument, soundness_score, verbal_feedback>], sorted by soundness_score desc
        logging.debug(f"ranking {question}; {ranking}")

        # construct new observations and update agents_observations_mapping
        for i, mapping in enumerate(agents_observations_mapping):
            agent, assigned_ans, initial_conf, count = mapping
            original_observations = (
                f"Your original answer is {assigned_ans}, with a confidence of {initial_conf:.2f}"
            )
            
            for rank, (ans, argument, soundness, feedback) in enumerate(ranking):
                general_feedback = (
                    f"'''{argument}'''\n, which received the following rating and feedback from other deliberators:"
                    f"Soundness score: {soundness:.2f} (ranked {rank+1} out of {len(ranking)})\n"
                    f"Feedback: {feedback}"
                )
                if ans == assigned_ans:
                    feedback_supporting = f"An argument supporting your original answer is\n{general_feedback}"
                    if not self_popularity:
                        feedback_supporting += f"\nNote that {count-1} other {'person' if count == 2 else 'people'} (out of {m}) also agreed with you."
                else:
                    feedback_opposing = f"An argument from the opposing side is\n{general_feedback}"
                    if not self_popularity:
                        feedback_opposing += f"\nNote {m - count} {'person' if m - count == 1 else 'people'} disagreed with you."                   
                    
            self_estimate_popularity = f"Based on the evidence presented, estimate how many deliberators (including yourself, out of {m}) are on your side." if self_popularity else ""
            new_observations = f"Recall that your original confidence was {initial_conf:.2f}\n{feedback_opposing}\n{feedback_supporting}\n{self_estimate_popularity}"
            agents_observations_mapping[i] = tuple((agent, initial_conf, original_observations, new_observations))

    # re-voting with new observations (and the corresponding ranking/feedback, if no early consensus)
    final_votes_raw = revote(question, agents_observations_mapping)
    rationales = [vote[-1] for vote in final_votes_raw]
    final_set = construct_stances(
        final_votes_raw, question, nli_classifier, nli_tokenizer, conf_rationales=rationales
    )
    return arguments, final_votes_raw, final_set


def save_vote_history(
    question_id: str,
    original_votes: List[Any],
    original_stance_list: List[Tuple[str, float, float | None, int, str]],
    final_votes: List[Tuple[str, str, str, float, float | None, str]],
    final_stance_list: List[Tuple[str, float, float | None, int, str]],
    final_majority: Tuple[str, float, int, str],
    output_filepath: Path,
    dataset: str,
):
    vote_keys = ["agent_id", "model", "answer", "verbal_confidence", "sequence_probability", "confidence_metrics"]
    stance_keys = ["answer_class", "avg_verbal_confidence", "avg_sequence_probability", "count", "rationale"]

    original_votes_with_keys = [dict(zip(vote_keys, original_vote)) for original_vote in original_votes]    
    original_stance_list_with_keys = [dict(zip(stance_keys, stance)) for stance in original_stance_list]
    
    vote_keys[-1] = "rationale"
    final_votes_with_keys = [dict(zip(vote_keys, final_vote)) for final_vote in final_votes]
    final_stance_list_with_keys = [dict(zip(stance_keys, stance)) for stance in final_stance_list]

    res = {
        "qid": question_id,
        "original_votes": original_votes_with_keys,
        "original_stances": original_stance_list_with_keys,
        "final_votes": final_votes_with_keys,
        "final_stances": final_stance_list_with_keys,
        "final_majority_ans": final_majority[0],
        "final_verbal_confidence": final_majority[1],
    }
    with open(f"{str(output_filepath)}/{dataset}.jsonl", mode="a") as fp:
        fp.write(json.dumps(res, indent=1) + "\n")

def save_agent_info(agent: BaseAgent, dirpath: str = "data/memory/agents/"):
    agent_info = agent.get_agent_parameters()
    api_call = agent.callback_handlers[0]
    api_call_info = dict(
        {
            "Number of successful_requests": api_call.successful_requests,
            "Number of total token": api_call.total_tokens,
            "Number of prompt token": api_call.prompt_tokens,
            "Number of completion token": api_call.completion_tokens,
            "Total cost for this agent": api_call.total_cost,
        }
    )
    output_filepath = f"{dirpath}/info_{agent.id}.json"
    Path(output_filepath).mkdir(parents=True, exist_ok=True)
    with open(f"{str(output_filepath)}/info_{agent.id}.json", "w+") as outfile:
        json.dump(api_call_info | agent_info, outfile)

def agents_deliberation_single_thread(
    df: pd.DataFrame,
    expert_agent_pool: Dict[str, Any],
    general_agent_pool: List[Any],
    nli_tokenizer: Any,
    nli_classifier: Any,
    output_filepath: Path,
    dataset: str,
    long_form: bool = False,
):
    if not nli_tokenizer or not nli_classifier:
        nli_tokenizer, nli_classifier = load_entailment_classifier()
    for _, record in tqdm(df.iterrows()):
        # Stage 1
        original_votes, stances = stance_generation(record, expert_agent_pool, nli_tokenizer, nli_classifier)
        if not len(stances):
            logging.info(f"Skip {record['qid']}")
            continue
        original_stance_list = sorted(stances, key=lambda t: t[3])
        # Stage 2
        verify = len(set(["pot", "search"]).intersection(list(expert_agent_pool.keys()))) == 0
        arguments, final_votes, final_ans_set = deliberate_with_feedback(
            record["question"], general_agent_pool, nli_tokenizer, nli_classifier, original_stance_list, long_form=long_form, verify=verify, 
        )
        for i, ans_cls in enumerate(original_stance_list):
            if not ans_cls[-1]:
                original_stance_list[i][-1] = arguments[ans_cls[0]]

        final_majority = sorted(list(final_ans_set), key=lambda t: t[3])[-1]
        logging.debug(f"final majority vote: {final_majority}")
        save_vote_history(
            record["qid"],
            original_votes,
            original_stance_list,
            final_votes,
            final_ans_set,
            final_majority,
            output_filepath,
            dataset,
        )

        for agent in list(itertools.chain(*expert_agent_pool.values())) + general_agent_pool:
            if agent.model_type in TEST_API_MODELS["openai-chat"]:
                save_agent_info(agent)


def allocate_slots(model_ensemble: bool, group_size: int, validation_data: pd.DataFrame, use_vllm: bool = False):
    # if using model_ensemble (k models), allocation_size = size(expert_agents) // k
    if model_ensemble:
        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("Cohere API not found.")
        if not os.getenv("HF_TOKEN"):
            raise ValueError("Huggingface access token for Llama not found.")   
        if not all(
            model.split("/")[-1] in itertools.chain(*TEST_API_MODELS.values()) or model.split("/")[-1] in itertools.chain(*TEST_HF_MODELS.values())
            for model in MODEL_ENSEMBLE_CHOICES
        ):
            raise ValueError("Model not supported.")
        allocation_size = group_size // len(MODEL_ENSEMBLE_CHOICES)
    else:
        allocation_size = group_size

    return allocate_agent_slots(validation_data, allocation_size, use_vllm=use_vllm)


def main(args):
    logging_level = logging.INFO if args.logging_level == "info" else logging.DEBUG
    logging.basicConfig(
        filename=args.logfile_path,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging_level,
    )
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logging.debug(f"device: {device}")
    # Load environment variables from .env file
    load_dotenv(args.api_key_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    test_data, dev_data = sample_input_data(args.input_dataset, args.test_sample_size, args.dev_sample_size)

    # Stage 1 agents
    if args.agent_ensemble and not args.long_form:
        # auto-select
        slots_allocation = allocate_slots(args.model_ensemble, args.group_size, dev_data, use_vllm=args.vllm)
        expert_agent_pool = populate_expert_agents(slots_allocation, args.model_ensemble, use_vllm=args.vllm)
    else:
        # long-form generation (with model_ensemble) or abalation
        model_ensemble = True if args.long_form else args.model_ensemble
        group_size = args.group_size // len(MODEL_ENSEMBLE_CHOICES) if model_ensemble else args.group_size
        # currently CoT/ZS prompting in half-half for long-form generation
        expert_agent_pool = populate_expert_agents(dict({"gen": group_size//2, "cot": group_size//2}), model_ensemble, use_vllm=args.vllm)

    # Stage 2 agents (same size as Stage 1 agents), fixed for all queries
    general_agent_pool = populate_general_agents(args.group_size)    
    logging.debug(f"expert agents: {expert_agent_pool}\ngeneral agents: {general_agent_pool}")

    # nli_tokenizer, nli_classifier = load_entailment_classifier()
    nli_tokenizer, nli_classifier = None, None

    df_list = np.array_split(test_data, args.n_thread)
    Path(args.memory_filepath).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.n_thread) as executor:
        futures = [
            executor.submit(
                agents_deliberation_single_thread,
                df,
                expert_agent_pool,
                general_agent_pool,
                nli_tokenizer,
                nli_classifier,                
                Path(args.memory_filepath),
                args.input_dataset,
                args.long_form,
            )
            for df in df_list
        ]
        for future in as_completed(futures):
            logging.info(future.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-logfile_path",
        default="data/logs.log",
        type=str,
        help="logging ouput file",
    )
    parser.add_argument(
        "-logging_level",
        default="info",
        choices=["debug", "info"],
        type=str,
        help="console logging level",
    )
    parser.add_argument(
        "-api_key_path",
        default=".env",
        type=str,
        help="path to the env file with all api keys",
    )
    parser.add_argument(
        "-memory_filepath",
        default="data/memory/output/",
        type=str,
        help="path to the simulation output",
    )
    parser.add_argument(
        "--agent_ensemble",
        default=True,  # no agent_ensemble for long-form generation tasks
        action="store_true",
        help="whether to ensemble with multiple specialized agents",
    )
    parser.add_argument(
        "--model_ensemble",
        default=False,
        action="store_true",
        help="whether to ensemble with multiple backbone models",
    )
    parser.add_argument(
        "--long_form",
        default=False,  # for long-form generation: do model_ensemble but not agent_ensemble (half ZS half CoT for now)
        action="store_true",
        help="whether the task is long-form generation",
    )
    parser.add_argument(
        "--vllm",
        default=False,
        action="store_true",
        help="whether to use vllm to speed up inference",
    )
    parser.add_argument(
        "-n_thread",
        type=int,
        default=1,
        required=False,
        help="number of threads",
    )
    parser.add_argument(
        "-group_size",
        type=int,
        default=6,
        required=False,
        help="number of expert agents in self-deliberation",
    )
    parser.add_argument(
        "-input_dataset",
        type=str,
        choices=["triviaqa-dev", "sciq-valid", "truthfulqa", "math-test-prm800k", "gsm8k-test", "WikiLingua-1000-chn-eng", "theoremqa-test", "ambigqa", "gpqa_diamond", "dateUnd", "prfLaw", "Biz-Ethics"],
        help="name of the input dataset",
    )
    parser.add_argument(
        "-dev_sample_size",
        required=False,
        default=1,
        type=int,
        help="size of the sampled development set for agent allocation",
    )
    parser.add_argument(
        "-test_sample_size",
        required=True,
        type=int,
        help="total number of examples to sample for the test set",
    )

    args = parser.parse_args()
    main(args)
