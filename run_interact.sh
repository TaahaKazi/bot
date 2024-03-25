#! /bin/bash

MODEL=gpt-3.5-turbo
python interact.py --model_name $MODEL --faiss_db mw-context-2-20perdomain.vec --num_examples 2 --database_path multiwoz_database --context_size 2 --dataset multiwoz --ontology ontology.json --run_name wasd
