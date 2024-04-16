"""Load pretrained models from Huggingface and do inference"""
from typing import Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from vllm import LLM, SamplingParams
from huggingface_hub import login
import torch
import numpy as np
import logging

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DEFAULT_CACHE_DIR = "data/hf_cache"
DEFAULT_NLI_CLS = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def load_entailment_classifier(nli_model: str = DEFAULT_NLI_CLS, cache_dir: str = DEFAULT_CACHE_DIR):
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model, use_fast=True, cache_dir=cache_dir)
    nli_classifier = AutoModelForSequenceClassification.from_pretrained(nli_model, cache_dir=cache_dir)
    logging.debug("entailment_classifier loaded")
    return nli_tokenizer, nli_classifier

def load_causal_lm(model_id: str, cache_dir: str = DEFAULT_CACHE_DIR, access_token: Any = None, use_vllm: bool = False):
    if "llama" in model_id.lower():
        assert access_token, "HF access token required."
    if use_vllm:
        # known issue: concurrent inference
        # https://github.com/vllm-project/vllm/issues/1285
        # https://github.com/vllm-project/vllm/issues/1200
        # e.g Llama-2-13b-chat-hf, Mistral-7B-Instruct-v0.1, vicuna-13b-v1.3
        login(token=access_token)
        logging.debug(GenerationConfig.from_pretrained(model_id))
        vllm_model = LLM(model=model_id, trust_remote_code=True, download_dir=cache_dir)
        logging.debug(f"{model_id} loaded with {vllm_model}")
        return None, vllm_model
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, 
                                              cache_dir=cache_dir, 
                                              use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config, 
                                                    cache_dir=cache_dir, 
                                                    use_auth_token=access_token)   
        logging.debug(f"{model_id} loaded")
        return tokenizer, model

def causal_lm_generate(tokenizer: Any, model: Any, prompt: str, max_new_tokens: int = 256, return_joint_prob: bool = True, use_hf_template: bool = False):
    lm_prob = None
    if not tokenizer:
        # vllm TODO:needs template
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.6, top_p=0.9, top_k=50, logprobs=1)  # same params with HF
        res = model.generate(prompt, sampling_params)[0]
        logging.debug(f"vllm prompt: {res.prompt}")
        output = res.outputs[0]
        lm_prob = np.exp(output.cumulative_logprob / len(output.token_ids))
        output_text = output.text
        logging.debug(f"vllm: {model} output length: {len(output.token_ids)}, logits: {output.cumulative_logprob}, normalized lm_prob: {lm_prob}, text: {output_text}")        
    else:
        if use_hf_template:
            tokenized_chat = tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt")
            model_inputs = tokenized_chat.to(device)
            input_length = model_inputs.shape[1]
            output = model.generate(tokenized_chat, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=return_joint_prob)
        else:
            model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = model_inputs.input_ids.shape[1]
            output = model.generate(**model_inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=return_joint_prob)

        generated_tokens = output.sequences[:, input_length:][0]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if return_joint_prob:
            transition_scores = model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            ),
            # for tok, score in zip(generated_tokens, transition_scores[0]):
            #     logging.debug(f"| {tok} | {tokenizer.decode(tok)} | {score.cpu().numpy()} | {np.exp(score.cpu().numpy())}")
            token_probs = np.exp(torch.squeeze(transition_scores[0], dim=0).cpu()).tolist()
            # if isinstance(token_probs, float):
            #     token_probs = [token_probs]
            logging.debug(f"transition_scores[0]: {torch.squeeze(transition_scores[0], dim=0)}, {transition_scores[0].size()}, token_probs: {token_probs}, {len(token_probs)}")
            lm_prob = np.prod(token_probs) ** (1 / len(token_probs))  #  P(y_1,...,y_n)^(1/n)
            logging.debug(f"{model.model} output text: {str(output_text).strip()}, length: {len(token_probs)}, joint_prob_with_length_penalty: {lm_prob:.3f}")
    return lm_prob, str(output_text).strip()
    
        

