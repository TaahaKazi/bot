import argparse
import pickle
import json
import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from pynvml import *
from datasets import load_dataset
import wandb
import logging
import transformers

from model import (
    FewShotPromptedLLM,
    SimplePromptedLLM,
    FewShotOpenAILLM,
    ZeroShotOpenAILLM,
    FewShotOpenAIChatLLM,
    ZeroShotOpenAIChatLLM,
    FewShotAlpaca,
    ZeroShotAlpaca
    )
from loaders import load_mwoz, load_sgd
from delex import prepareSlotValuesIndependent, delexicalise, delexicaliseReferenceNumber
from definitions import MW_FEW_SHOT_DOMAIN_DEFINITIONS, MW_ZERO_SHOT_DOMAIN_DEFINITIONS, SGD_FEW_SHOT_DOMAIN_DEFINITIONS, SGD_ZERO_SHOT_DOMAIN_DEFINITIONS

from database import MultiWOZDatabase
from utils import parse_state, ExampleRetriever, ExampleFormatter, print_gpu_utilization, SGDEvaluator
from mwzeval.metrics import Evaluator as MWEvaluator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

transformers.set_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/home/hudecek/hudecek/hf_cache")
    parser.add_argument("--model_name", type=str, default="allenai/tk-instruct-3b-def-pos-neg-expl")
    parser.add_argument("--faiss_db", type=str, default="multiwoz-context-db.vec")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--dials_total", type=int, default=100)
    parser.add_argument("--database_path", type=str, default="multiwoz_database")
    parser.add_argument("--dataset", type=str, default="multiwoz")
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--ontology", type=str, default="ontology.json")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--use_gt_state", action='store_true')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--use_zero_shot", action='store_true')
    args = parser.parse_args()
    config = {
        "model_name": args.model_name,
        "faiss_db": args.faiss_db,
        "num_examples": args.num_examples,
        "dataset": args.dataset,
        "context_size": args.context_size,
        "use_gt_state": args.use_gt_state,
        "use_zero_shot": args.use_zero_shot,
        "split": args.split,
        "num_dialogs": args.dials_total,
    }
    wandb.init(project='llmbot', entity='hlava', config=config)
    if 'tk-instruct-3b' in args.model_name:
        model_name = 'tk-3B'
    elif 'tk-instruct-11b' in args.model_name:
        model_name = 'tk-11B'
    elif 'opt-iml-1.3b' in args.model_name:
        model_name = 'opt-iml-1.3b'
    elif 'opt-iml-30b' in args.model_name:
        model_name = 'opt-iml-30b'
    elif 'NeoXT' in args.model_name:
        model_name = 'GPT-NeoXT-20b'
    elif 'gpt-3.5' in args.model_name:
        model_name = 'ChatGPT'
    elif args.model_name == 'alpaca':
        model_name = 'Alpaca-LoRA'
    else:
        model_name = 'GPT3.5'
    wandb.run.name = f'{args.run_name}-{args.dataset}-{model_name}-examples-{args.num_examples}-ctx-{args.context_size}'
    report_table = wandb.Table(columns=['id', 'context', 'raw_state', 'parsed_state', 'response'])
    if args.model_name.startswith("text-"):
        model_factory = ZeroShotOpenAILLM if args.use_zero_shot else FewShotOpenAILLM
        model = model_factory(args.model_name)
    elif args.model_name.startswith("gpt-"):
        model_factory = ZeroShotOpenAIChatLLM if args.use_zero_shot else FewShotOpenAIChatLLM
        model = model_factory(args.model_name)
    elif any([n in args.model_name for n in ['opt', 'NeoXT']]):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                    low_cpu_mem_usage=True,
                                                    cache_dir=args.cache_dir,
                                                    device_map="auto",
                                                    load_in_8bit=True)
        model_factory = SimplePromptedLLM if args.use_zero_shot else FewShotPromptedLLM
        model = model_factory(model, tokenizer, type="causal")
    elif 'alpaca' in args.model_name:
        model_factory = ZeroShotAlpaca if args.use_zero_shot else FewShotAlpaca
        model = model_factory(model_name="Alpaca-LoRA")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                    low_cpu_mem_usage=True,
                                                    cache_dir=args.cache_dir,
                                                    device_map="auto",
                                                    load_in_8bit=True)
        model_factory = SimplePromptedLLM if args.use_zero_shot else FewShotPromptedLLM
        model = model_factory(model, tokenizer, type="seq2seq")

    with open(args.faiss_db, 'rb') as f:
        faiss_vs = pickle.load(f)
    with open(args.ontology, 'r') as f:
        ontology = json.load(f)
    if args.dataset == 'multiwoz':
        database = MultiWOZDatabase(args.database_path)
        with open('multiwoz-state-update-1turn-only-ctx2.vec', 'rb') as f:
            state_vs = pickle.load(f)
        delex_dic = prepareSlotValuesIndependent(args.database_path)
    else:
        state_vs = faiss_vs
        delex_dic = None
    example_retriever = ExampleRetriever(faiss_vs)
    state_retriever = ExampleRetriever(state_vs)
    example_formatter = ExampleFormatter(ontology=ontology)

    history = []
    n = 0
    results = {}
    results_wo_state = {}
    last_dial_id = None
    total = args.dials_total
    if args.dataset == 'multiwoz':
        data_gen = load_mwoz(args.database_path, args.context_size, split=args.split, total=total, shuffle=False)
    else:
        data_gen = load_sgd(args.context_size, split=args.split, total=total, shuffle=True)
    tn = 0
    progress_bar = tqdm.tqdm(total=total)
    for it, turn in enumerate(data_gen):
        if last_dial_id != turn['dialogue_id']:
            last_dial_id = turn['dialogue_id']
            n += 1
            progress_bar.update(1)
            tn = 0
            if n > total:
                break
            history = []
            dialogue_id = turn['dialogue_id']
            results[dialogue_id] = []
            results_wo_state[dialogue_id] = []
            total_state = {}
            print('=' * 100)
            previous_domain = None
        tn += 1
        question = turn['question']
        gold_response = turn['metadata']['response']
        gt_state = turn['gt_state']
        if len(gt_state) == 0:
            gt_state = {}
        new_gt_state = {}
        for domain, ds in gt_state.items():
            for sl, val in ds.items():
                if domain not in new_gt_state:
                    new_gt_state[domain] = {sl: val}
                else:
                    new_gt_state[domain][sl] = val
        retrieve_history = history + ["Customer: " + question]
        retrieved_examples = example_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=5)
        retrieved_domains = [example['domain'] for example in retrieved_examples]
        selected_domain = Counter(retrieved_domains).most_common(1)[0][0]
        if previous_domain != selected_domain:
           #  total_state = {}
            previous_domain = selected_domain
        retrieved_examples = [example for example in retrieved_examples if example['domain'] == selected_domain]
        state_examples = [example for example in state_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=7) if example['domain'] == selected_domain]
        num_examples = min(len(retrieved_examples), args.num_examples)
        positive_state_examples = example_formatter.format(state_examples[:num_examples],
                                                           input_keys=["context"],
                                                           output_keys=["state"])
        negative_state_examples = example_formatter.format(state_examples[:num_examples],
                                                           input_keys=["context"],
                                                           output_keys=["state"],
                                                           corrupt_state=True)
        response_examples = example_formatter.format(retrieved_examples[:num_examples],
                                                    input_keys=["context", "state", "database"],
                                                    output_keys=["response"])
        
        if args.dataset == 'multiwoz':
            domain_definition = MW_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain] if args.use_zero_shot else MW_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
            available_domains = list(MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        else:
            domain_definition = SGD_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain] if args.use_zero_shot else SGD_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
            available_domains = list(SGD_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        state_prompt = domain_definition.state_prompt
        response_prompt = domain_definition.response_prompt
        
        if args.use_gt_state:
            state = str(new_gt_state)
            parsed_state = total_state = final_state = new_gt_state
        else:
            try:
                kwargs = {
                    "history": "\n".join(history),
                    "utterance": question.strip()
                }
                if not args.use_zero_shot:
                    kwargs["positive_examples"] = positive_state_examples
                    kwargs["negative_examples"] = negative_state_examples
                state, filled_state_prompt = model(state_prompt, predict=True, **kwargs)
                if n < 5:
                    print("Filled prompt:", filled_state_prompt)
            except:
                state = "{}"

            parsed_state = parse_state(state, default_domain=selected_domain)
            if selected_domain not in parsed_state:
                parsed_state[selected_domain] = {}
            if not isinstance(parsed_state[selected_domain], dict):
                parsed_state[selected_domain] = {}
            keys_to_remove = [k for k in parsed_state[selected_domain].keys() if k not in domain_definition.expected_slots]
            for k in keys_to_remove:
                del parsed_state[selected_domain][k]
            try:
                for domain, ds in parsed_state.items():
                    for slot, value in ds.items():
                        pass
            except:
                parsed_state = {domain: {}}
            
            final_state = {}
            for domain, ds in parsed_state.items():
                if domain in available_domains:
                    final_state[domain] = ds
            
            for domain, dbs in final_state.items():
                if domain not in total_state:
                    total_state[domain] = dbs
                else:
                    for slot, value in dbs.items():
                        value = str(value)
                        if value not in ['dontcare', 'none', '?', ''] and len(value) > 0:
                            total_state[domain][slot] = value
        
        print('-' * 100)
        print(f"Question: {question}", flush=True)
        print(f"Selected domain: {selected_domain}", flush=True)
        logger.info(f"Raw State: {state}")
        print(f"Raw State: {state}", flush=True)
        logger.info(f"Parsed State: {final_state}")
        print(f"Parsed State: {final_state}", flush=True)
        logger.info(f"Total State: {total_state}")
        print(f"Total State: {total_state}", flush=True)

        if args.dataset == 'multiwoz':
            database_results = {domain: len(database.query(domain=domain, constraints=ds))
                                for domain, ds in total_state.items() if len(ds) > 0}
        else:
            database_results = turn['metadata']['database']
        logger.info(f"Database Results: {database_results}")
        print(f"Database Results: {database_results}", flush=True)
        
        try:
            kwargs = {
                "history": "\n".join(history),
                "utterance": question.strip(),
                "state": json.dumps(total_state).replace("{", '<').replace("}", '>'),
                "database": str(database_results)
            }
            if not args.use_zero_shot:
                kwargs["positive_examples"] = response_examples
                kwargs["negative_examples"] = []

            response, filled_prompt = model(response_prompt, predict=True, **kwargs)
            if n < 5:
                print("Filled response prompt:", filled_prompt)
        except:
            response = ''

        if args.dataset == 'multiwoz':
            response = delexicalise(response, delex_dic)
            response = delexicaliseReferenceNumber(response)
        
        logger.info(f"Response: {response}")
        print(f"Response: {response}", flush=True)
        print(f"Gold Response: {gold_response}", flush=True)

        history.append("Customer: " + question)
        report_table.add_data(f"{dialogue_id}-{tn}", " ".join(history), state, json.dumps(final_state), response)
        history.append("Assistant: " + gold_response)
        
        results[dialogue_id].append({
            "domain": selected_domain,
            "active_domains": [selected_domain],
            "response": response,
            "state": final_state,
        })
        results_wo_state[dialogue_id].append({
            "domain": selected_domain,
            "active_domains": [selected_domain],
            "response": response,
        })
    wandb.log({"examples": report_table})
    progress_bar.close()

    if args.dataset == 'multiwoz':
        evaluator = MWEvaluator(bleu=True, success=True, richness=True, jga=True, dst=True)
        eval_results = evaluator.evaluate(results)
        for metric, values in eval_results.items():
            if values is not None:
                for k, v in values.items():
                    wandb.log({f"MW_{metric}-{k.ljust(15)}": v})

        evaluator = MWEvaluator(bleu=True, success=True, richness=True)
        eval_results = evaluator.evaluate(results_wo_state)
        for metric, values in eval_results.items():
            if values is not None:
                for k, v in values.items():
                    wandb.log({f"MW_GT_{k.ljust(15)}": v})
    else:
        evaluator = SGDEvaluator(split=args.split)
        metrics = {}
        metrics.update(evaluator.get_bleu(results))
        metrics.update(evaluator.get_eval(results))
        for metric, val in metrics.items():
            wandb.log({f"SGD_{metric}": val})
