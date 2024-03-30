import sys
import pickle
import json
import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset
import wandb
import logging
import transformers
import random

from modelling import (
    FewShotPromptedLLM,
    SimplePromptedLLM,
    FewShotOpenAILLM,
    ZeroShotOpenAILLM,
    FewShotOpenAIChatLLM,
    ZeroShotOpenAIChatLLM,
    FewShotAlpaca,
    ZeroShotAlpaca
)

from delex import prepareSlotValuesIndependent, delexicalise, delexicaliseReferenceNumber
from definitions import MW_FEW_SHOT_DOMAIN_DEFINITIONS, MW_ZERO_SHOT_DOMAIN_DEFINITIONS, SGD_FEW_SHOT_DOMAIN_DEFINITIONS, SGD_ZERO_SHOT_DOMAIN_DEFINITIONS, multiwoz_domain_prompt, sgd_domain_prompt

from database import MultiWOZDatabase
from utilities import parse_state, ExampleRetriever, ExampleFormatter, print_gpu_utilization, SGDEvaluator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

transformers.set_seed(42)

class TODSystem:
    def __init__(self, cache_dir, model_name, faiss_db, num_examples, dials_total, database_path, dataset, context_size, ontology, output, run_name, use_gt_state, use_gt_domain, use_zero_shot, verbose, goal_data, debug):
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.faiss_db = faiss_db
        self.num_examples = num_examples
        self.dials_total = dials_total
        self.database_path = database_path
        self.dataset = dataset
        self.context_size = context_size
        self.ontology = ontology
        self.output = output
        self.run_name = run_name
        self.use_gt_state = use_gt_state
        self.use_gt_domain = use_gt_domain
        self.use_zero_shot = use_zero_shot
        self.verbose = verbose
        self.goal_data = goal_data
        self.debug = debug
        # Initialize other necessary components here, such as model loading

        self.config = {
            "model_name": self.model_name,
            "faiss_db": self.faiss_db,
            "num_examples": self.num_examples,
            "dataset": self.dataset,
            "context_size": self.context_size,
            "use_gt_state": self.use_gt_state,
            "use_zero_shot": self.use_zero_shot,
            "use_gt_domain": self.use_gt_domain,
        }
        if 'tk-instruct-3b' in self.model_name:
            self.model_name = 'tk-3B'
        elif 'tk-instruct-11b' in self.model_name:
            self.model_name = 'tk-11B'
        elif 'opt-iml-1.3b' in self.model_name:
            self.model_name = 'opt-iml-1.3b'
        elif 'opt-iml-30b' in self.model_name:
            self.model_name = 'opt-iml-30b'
        elif 'NeoXT' in self.model_name:
            self.model_name = 'GPT-NeoXT-20b'
        elif 'gpt-3.5' in self.model_name:
            self.model_name = 'ChatGPT'
        elif self.model_name == 'alpaca':
            self.model_name = 'Alpaca-LoRA'
        else:
            self.model_name = 'GPT3.5'

        # TODO check this logic
        self.model_name = model_name

        if model_name.startswith("text-"):
            model_factory = ZeroShotOpenAILLM if self.use_zero_shot else FewShotOpenAILLM
            self.model = model_factory(self.model_name)
            self.domain_model = ZeroShotOpenAILLM(self.model_name)
        elif model_name.startswith("gpt-"):
            model_factory = ZeroShotOpenAIChatLLM if self.use_zero_shot else FewShotOpenAIChatLLM
            self.model = model_factory(self.model_name)
            self.domain_model = ZeroShotOpenAIChatLLM(self.model_name)
        elif any([n in self.model_name for n in ['opt', 'NeoXT']]):
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            model_w = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                           low_cpu_mem_usage=True,
                                                           cache_dir=self.cache_dir,
                                                           device_map="auto",
                                                           load_in_8bit=True)
            model_factory = SimplePromptedLLM if self.use_zero_shot else FewShotPromptedLLM
            self.model = model_factory(model_w, tokenizer, type="causal")
            self.domain_model = SimplePromptedLLM(model_w, tokenizer, type="causal")
        elif 'alpaca' in self.model_name:
            model_factory = ZeroShotAlpaca if self.use_zero_shot else FewShotAlpaca
            self.model = model_factory(model_name="Alpaca-LoRA")
            self.domain_model = ZeroShotAlpaca(model_name="Alpaca-LoRA")
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            model_w = AutoModelForSeq2SeqLM.from_pretrained(self.model_name,
                                                            low_cpu_mem_usage=True,
                                                            cache_dir=self.cache_dir,
                                                            device_map="auto",
                                                            load_in_8bit=True)
            model_factory = SimplePromptedLLM if self.use_zero_shot else FewShotPromptedLLM
            self.model = model_factory(model_w, tokenizer, type="seq2seq")
            self.domain_model = SimplePromptedLLM(model_w, tokenizer, type="seq2seq")

        with open(self.faiss_db, 'rb') as f:
            self.faiss_vs = pickle.load(f)
        with open(self.ontology, 'r') as f:
            self.ontology = json.load(f)
        if self.dataset == 'multiwoz':
            self.domain_prompt = multiwoz_domain_prompt
            self.database = MultiWOZDatabase(self.database_path)
            self.state_vs = self.faiss_vs
            self.delex_dic = prepareSlotValuesIndependent(self.database_path)
        else:
            self.domain_prompt = sgd_domain_prompt
            self.state_vs = self.faiss_vs
            self.delex_dic = None
        self.example_retriever = ExampleRetriever(self.faiss_vs)
        self.state_retriever = ExampleRetriever(self.state_vs)
        self.example_formatter = ExampleFormatter(ontology=self.ontology)

        self.history = []
        self.last_dial_id = None
        self.total = self.dials_total
        self.dialogue_id = 1
        self.tn = 0
        self.total_state = {}


    def lexicalize(self, results, domain, response):
        # Method adapted from global lexicalize function
        if domain not in results:
            return response
        elif len(results[domain]) == 0:
            return response
        item = results[domain][0]
        extend_dct = {f"{domain}_{key}": val for key, val in item.items()}
        item.update(extend_dct)
        item.update({f"value_{key}": val for key, val in item.items()})
        item["choice"] = str(len(results[domain]))
        for key, val in item.items():
            x = f"[{key}]"
            if x in response:
                response = response.replace(x, val)
        return response

    def run(self, user_input=None):

        if self.debug:
            print("DEBUG MODE:")
        user_input = user_input.lower()
        self.tn += 1
        question = user_input
        retrieve_history = self.history + ["Customer: " + question]
        retrieved_examples = self.example_retriever.retrieve("\n".join(retrieve_history[-self.context_size:]), k=20)
        retrieved_domains = [example['domain'] for example in retrieved_examples]
        selected_domain, dp = self.domain_model(self.domain_prompt, predict=True, history="\n".join(self.history[-2:]),
                                           utterance=F"Customer: {question.strip()}")
        if self.dataset == 'multiwoz':
            available_domains = list(MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        else:
            available_domains = list(SGD_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        if self.verbose:
            pass
            # print(f"PREDICTED DOMAIN: {selected_domain}")
        if selected_domain not in available_domains:
            selected_domain = random.choice(available_domains)
        if self.dataset == 'multiwoz':
            domain_definition = MW_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain] if self.use_zero_shot else \
            MW_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
        else:
            domain_definition = SGD_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain] if self.use_zero_shot else \
            SGD_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
        retrieved_examples = [example for example in retrieved_examples if example['domain'] == selected_domain]
        num_examples = min(len(retrieved_examples), self.num_examples)
        num_state_examples = 5
        state_examples = [example for example in
                          self.state_retriever.retrieve("\n".join(retrieve_history[-self.context_size:]), k=20) if
                          example['domain'] == selected_domain][:num_state_examples]
        positive_state_examples = self.example_formatter.format(state_examples[:num_state_examples],
                                                           input_keys=["context"],
                                                           output_keys=["state"],
                                                           )
        # use_json=True)
        negative_state_examples = self.example_formatter.format(state_examples[:num_state_examples],
                                                           input_keys=["context"],
                                                           output_keys=["state"],
                                                           corrupt_state=True)
        response_examples = self.example_formatter.format(retrieved_examples[:num_examples],
                                                     input_keys=["context", "full_state", "database"],
                                                     output_keys=["response"],
                                                     use_json=True)

        state_prompt = domain_definition.state_prompt
        response_prompt = domain_definition.response_prompt

        try:
            kwargs = {
                "history": "\n".join(self.history),
                "utterance": question.strip()
            }
            if not self.use_zero_shot:
                kwargs["positive_examples"] = positive_state_examples
                kwargs["negative_examples"] = []  # negative_state_examples
            state, filled_state_prompt = self.model(state_prompt, predict=True, **kwargs)
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
            parsed_state = {selected_domain: {}}

        final_state = {}
        for domain, ds in parsed_state.items():
            if domain in available_domains:
                final_state[domain] = ds

        for domain, dbs in final_state.items():
            if domain not in self.total_state:
                self.total_state[domain] = dbs
            else:
                for slot, value in dbs.items():
                    value = str(value)
                    if value not in ['dontcare', 'none', '?', ''] and len(value) > 0:
                        self.total_state[domain][slot] = value

        if self.debug:
            print(f"Belief State: {self.total_state}", flush=True)

        if self.dataset == 'multiwoz':
            database_results = {domain: self.database.query(domain=domain, constraints=ds)
                                for domain, ds in self.total_state.items() if len(ds) > 0}
            for domain, rez in database_results.items():
                database_results[domain] = rez[:5]
        else:
            pass
        logger.info(f"Database Results: {database_results}")

        if self.debug:
            print(
            f"Database Results: {database_results[selected_domain][0] if selected_domain in database_results and len(database_results[selected_domain]) > 0 else 'EMPTY'}",
            flush=True)

        try:
            kwargs = {
                "history": "\n".join(self.history),
                "utterance": question.strip(),
                "state": json.dumps(self.total_state),  # .replace("{", '<').replace("}", '>'),
                "database": str(database_results)
            }
            if not self.use_zero_shot:
                kwargs["positive_examples"] = response_examples
                kwargs["negative_examples"] = []

            # response, filled_prompt = "IDK", "-"
            response, filled_prompt = self.model(response_prompt, predict=True, **kwargs)
            response = response.split("\n")[0]
        except:
            response = ''

        if self.dataset == 'multiwoz':
            pass

        self.history.append("Customer: " + question)
        self.history.append("Assistant: " + response)

        return response



if __name__ == "__main__":
    system = TODSystem(
        cache_dir="",
        model_name="gpt-3.5-turbo-instruct",
        faiss_db="multiwoz-context-db.vec",
        num_examples=2,
        dials_total=100,
        database_path="multiwoz_database",
        dataset="multiwoz",
        context_size=3,
        ontology="ontology.json",
        output="results",
        run_name="",
        use_gt_state=False,
        use_gt_domain=False,
        use_zero_shot=True,
        verbose=True,
        goal_data=None,
        debug=True,# Placeholder, adjust based on actual usage
    )
    response = system.run("Hi, I want to book a train from Cambridge to London")
    print(response)
    response = system.run("I want to travel on Tuesday")
    print(response)
    response = system.run("What is the travel time?")
    print(response)