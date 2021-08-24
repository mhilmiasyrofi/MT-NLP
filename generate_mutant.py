#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import pickle
from fluency_scorer import FluencyScorer
from Mutator import AnalogyMutator, ActiveMutator, create_sentence_candidates
import gensim.downloader as api
import pandas as pd
import argparse

import time

from multiprocessing import Pool, Process, Queue, Manager
import multiprocessing


def generate_mutant():

    args = get_args()
    sensitive_attribute = args.sensitive_attribute
    task = args.task
    print("=======")
    print("Sensitive attribute: ", sensitive_attribute)
    print("Task: ", task)
    print("=======")



    # load word2vec embedding
    print("word2vec model loading")
    word2vec = api.load("word2vec-google-news-300")
    print("word2vec model loaded")

    
    # sensitive_attribute = "gender"
    # sensitive_attribute = "country"
    

    ana = AnalogyMutator(sensitive_attribute, model=word2vec)
    act = ActiveMutator(sensitive_attribute)

    

    def mutate_text(sentence):
        output = create_sentence_candidates(
            sentence, ana, act)
        return output

    def execute_mut():
        '''for multiprocessing uaage'''
        while True:
            if not q.empty():
                label, text = q.get()
                output = mutate_text(text)
                
                if len(output) > 0:
                    original = output["original"]
                    label = [label] * len(output["original"])
                    mutant = output["mutant"]
                    template = output["template"]
                    identifier = output["identifier"]
                    type = output["type"]
                    q_to_store.put((
                        original, label, mutant, template, identifier, type
                    ))
            else:
                print("Finished")
                # return
                break

    # task = "twitter_s140"
    # task = "imdb"

    
    fpath = f"./asset/{task}/test.csv"

    df = pd.read_csv(fpath, names=["label", "sentence"], sep="\t")

    # df = df[:20]

    start = time.time()

    originals = []
    mutants = []
    labels = []
    templates = []
    identifiers = []
    types = []
    
    i = 0

    manager = multiprocessing.Manager()

    q = manager.Queue()
    q_to_store = manager.Queue()

    for index, row in df.iterrows():
        label = row["label"]
        text = row["sentence"]

        q.put((label, text))

    numList = []
    ## they use http://api.conceptnet.io for the mutation engine that limit the api request
    ## thus we can not use multiprocessing
    for i in range(1): 
        p = multiprocessing.Process(target=execute_mut, args=())
        numList.append(p)
        p.start()

    for i in numList:
        i.join()

    print("Generation Process finished.")

    while not q_to_store.empty():
        original, label, mutant, template, identifier, type = q_to_store.get()
        originals.extend(original)
        labels.extend(label)
        mutants.extend(mutant)
        templates.extend(template)
        identifiers.extend(identifier)
        types.extend(type)
        
    end = time.time()
    print("Execution Time: ", end-start)

    dm = pd.DataFrame(data={"label": labels, "mutant": mutants, "original": originals, "template": templates, "identifier": identifiers, "type":types})

    dm = dm.drop_duplicates()

    # ["label", "mutant", "original", "template","identifier", "type"]
    
    output_dir = f"./mutant/{sensitive_attribute}/{task}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dm.to_csv(output_dir + "test.csv", index=None, header=None, sep="\t")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensitive-attribute', default="gender", type=str)
    parser.add_argument('--task', default="imdb", type=str, help='dataset for generating mutants')
    
    return parser.parse_args()

if __name__ == "__main__" :
    generate_mutant()
