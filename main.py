import os
import sys
from tqdm import tqdm
import numpy as np
from time import sleep
import json
import csv
import argparse

from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from pyserini.search import get_topics, get_qrels

import openai


csv.field_size_limit(sys.maxsize)
number2word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
from collections import Counter

from utils import *

run_path = './runs_inter'
print(f'The working directory is {run_path}')


def model_init(comode):
    # model and index initilization
    print(f'{format_time()} load contriever ...')
    query_encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')

    print(f'{format_time()} initial lucene searcher ...')
    bm25_searcher = LuceneSearcher('indexes/lucene-index-msmarco-passage/')

    print(f'{format_time()} initial faiss searcher ...')
    searcher = FaissSearcher('indexes/contriever_msmarco_index/', query_encoder)
    print(f'{format_time()} model and index initilization finished ...')

    if comode=='rmdemo':
        print(f'{format_time()} load trec collections ...')
        psg_collections = load_tsv('data_msmarco/collection.tsv')
    else:
        psg_collections = None

    return query_encoder, bm25_searcher, searcher, psg_collections


def load_topics_qrels(eval_trec_mode):
    if 'dl20' in eval_trec_mode:
        topics = get_topics('dl20')
    else:
        topics = get_topics(f'{eval_trec_mode}')

    qrels = get_qrels(f'{eval_trec_mode}')
    return topics, qrels


def gen_query_embedding(all_emb_c, index_select, q_emb=None, ensemble=False):
    all_emb_c = np.take(all_emb_c, index_select, axis=0)
    
    if not ensemble:
        if q_emb is not None:
            all_emb_c = np.concatenate((all_emb_c, q_emb), axis=0) 


        avg_emb_c = np.mean(all_emb_c, axis=0)
        avg_emb_c = avg_emb_c.reshape((1, len(avg_emb_c)))
        
        return avg_emb_c
    
    else:
        q_emb_expand = np.repeat(q_emb, all_emb_c.shape[0], axis=0)
        emb_cat = np.stack([all_emb_c, q_emb_expand], axis=1)
        emb_cat = np.mean(emb_cat, axis=1, keepdims=True)
        return list(emb_cat)


def merge_hit(hits_list):
    docid2scores =[]
    for hits in hits_list:
        docid2score = {}
        for hit in hits:
            docid2score[hit.docid] = hit.score
        docid2scores.append(docid2score)

    sums = Counter()
    counters = Counter()
    for itemset in docid2scores:
        sums.update(itemset)
        counters.update(itemset.keys())

    hits_ret = {x: float(sums[x])/counters[x] for x in sums.keys()}
    hits_ret = dict(sorted(hits_ret.items(), key=lambda item: item[1], reverse = True)[:1000])
    return hits_ret


def call_codex_read_api(message, llm='chatgpt', beamsize=1, temperature=0.7, max_tokens=512):
    def parse_codex_result(result):
        to_return = []
        for idx, g in enumerate(result['choices']):
            text = g['text']
            logprob = sum(g['logprobs']['token_logprobs'])
            to_return.append((text, logprob))
        res = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
        return res

    def parse_chat_result(result):
        to_return = []
        for idx, g in enumerate(result['choices']):
            text = g['message']
            to_return.append((text['content']))
        return to_return

    num_try = 0
    get_result = False
    while not get_result:
        try:
            if llm=='chatgpt':
                result = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0301',
                    messages=message,
                    api_key='',  # FIXME HERE! change it to your own openai key
                    n=beamsize,
                )
            else:
                result = openai.Completion.create(
                    engine='gpt-3.5-turbo-0301', 
                    prompt='',
                    api_key='', 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    n=beamsize,
                    stop=['\n\n'],
                    logprobs=1
                )
            get_result = True
            num_try = 0
        except Exception as e:
            num_try += 1
            print(e)
            print(f'---------Failed!!! Try {num_try}----------')
            sleep(2)

    if llm=='chatgpt':
        return parse_chat_result(result)
    else:
        return parse_codex_result(result)



def gen_context_with_llm(args, topics, qrels, iter, prompt, demo_psgs=None, with_demo=False, load_ctx=False):

    qid2contexts = {}
    qid2suggquer = {}
    file_gpt_gen = f'{run_path}/{args.eval_trec_mode}-{args.llm}-gen-{args.prompt_prefix}-iter{iter}.jsonl'

    if os.path.exists(file_gpt_gen) and os.path.isfile(file_gpt_gen) and load_ctx:
        print(f'Read cache file: {file_gpt_gen}')
        with open(file_gpt_gen, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line.strip())
                qid2contexts[item['query_id']]=item['contexts']
                qid2suggquer[item['query_id']]=item['query_sug']
        print(f"{format_time()} Load llm ctx done ...")
    else:
        print(f'Write cache file: {file_gpt_gen}')
        with open(file_gpt_gen, 'w') as fgen:
            for qid in tqdm(topics, desc='Gen'):
                if qid in qrels: 
                    query = topics[qid]['title']
                    
                    if with_demo and demo_psgs is not None:
                        usr_prompt = prompt.format(query, demo_psgs[qid])
                    else:
                        usr_prompt = prompt.format(query)

                    if args.llm=='gpt3':
                        messages = usr_prompt
                    elif args.llm=='chatgpt':
                        sys_prompt = "You are a helpful assistant."
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": usr_prompt}
                        ]

                    contexts = [c.strip() for c in call_codex_read_api(messages, args.llm, args.num_ctx_gen)]

                    fgen.write(json.dumps({'query_id': qid, 'query': query, 'query_sug': [], 'contexts': contexts})+'\n')
                    qid2contexts[qid]=contexts
                    qid2suggquer[qid]=[]

    return qid2contexts, qid2suggquer


def ret_demo_with_context(args, topics, qrels, qid2contexts, searcher, use_context=True, ret_type='de2', eval=False, query_encoder=None):
    all_hits = []
    for id, qid in enumerate(tqdm(topics, desc='Ret')):
        if qid in qrels:
            query = topics[qid]['title']
            contexts = qid2contexts[qid][:args.num_ctx_use]

            if args.comode=='rmdemo':
            
                if 'bm2' in ret_type:   
                    query_exp = ' '.join(args.num_ctx_use*[query] + [psg.strip().replace('\n', ' ') for psg in contexts])
                    hits = bm25_searcher.search(query_exp, k=args.num_demo if not eval else 1000) 

                if 'de2' in ret_type:
                    q_emb = np.array(query_encoder.encode(query), ndmin=2) 
                    all_emb_c = [query_encoder.encode(c.strip().replace('/n', ' ')) for c in contexts] 
                    all_emb_c = np.array(all_emb_c) 
                    avg_emb = gen_query_embedding(all_emb_c, [0,1,2], q_emb)
                    hits = searcher.search(avg_emb, k=args.num_demo if not eval else 1000)
                
                all_hits.append(hits)

    return all_hits



def eval_trec(all_qids, all_hits, eval_trec_mode, eval_prefix='bm25'):
    with open(f'{run_path}/{eval_trec_mode}-{eval_prefix}-top1000-trec', 'w')  as f:
        for qid, hits in zip(all_qids, all_hits):
            rank=0
            for hit in hits:
                rank += 1
                f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')

    os.system(f'python -m pyserini.eval.trec_eval -c -l 2 -m map {eval_trec_mode} {run_path}/{eval_trec_mode}-{eval_prefix}-top1000-trec')
    os.system(f'python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {eval_trec_mode} {run_path}/{eval_trec_mode}-{eval_prefix}-top1000-trec')
    os.system(f'python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 {eval_trec_mode} {run_path}/{eval_trec_mode}-{eval_prefix}-top1000-trec')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--eval_bm25", action="store_true")
    parser.add_argument("--load_ctx", action="store_false")
    parser.add_argument("--num_ctx_gen", type=int, default=15) 
    parser.add_argument("--num_ctx_use", type=int, default=4)
    parser.add_argument("--num_demo", type=int, default=15)
    parser.add_argument("--llm", type=str, default="chatgpt", choices=["chatgpt", "gpt3"])
    parser.add_argument("--prompt_prefix", type=str, default="pro2")
    parser.add_argument("--comode", type=str, default="rmdemo")
    parser.add_argument("--demo_type", type=str, default="")
    parser.add_argument("--eval_trec_mode", type=str, default='dl19-passage')

    args = parser.parse_args()

    model_prefix = f'{args.num_ctx_use}ctx-{args.comode}'
    demo_type = f'-{args.num_demo}bm2-p1' if 'demo' in args.comode else ''
    
    # -------------------------------------------------------------------------------------------------------------------------
    setup_seed(args.seed)
    query_encoder, bm25_searcher, con_searcher, psg_collections = model_init(args.comode)
    topics, qrels = load_topics_qrels(args.eval_trec_mode)
    all_qids = [qid for qid in tqdm(topics) if qid in qrels]
    
    # -------------------------------------------------------------------------------------------------------------------------
    if args.eval_bm25:
        all_hits = [bm25_searcher.search(topics[qid]['title'], k=1000) for qid in tqdm(topics) if qid in qrels]
        eval_trec(all_qids, all_hits, args.eval_trec_mode, eval_prefix='bm25')
    # -------------------------------------------------------------------------------------------------------------------------

    num_iter = 2
    demo_psgs = None
    for iter in range(0, num_iter): 
        if iter==0:
            usr_prompt = "Please write a passage to answer the question\nQuestion: {}\nPassage:"
        else:
            usr_prompt = "Give a question \"{}\" and its possible answering passages \n{}\nplease write a correct answering passage." 

        # ----------------------------------------------------------------------------------------------------------
        print(f"{format_time()} Iter-{iter}: Generating context with LLM")
        qid2contexts, _ = gen_context_with_llm(args, topics, qrels, iter, usr_prompt, demo_psgs, with_demo=iter>0, load_ctx=True)

        # ----------------------------------------------------------------------------------------------------------
        if iter>0:
            print(f"{format_time()} Iter-{iter}: Evaluating on DL Trec with BM25")
            all_hits = ret_demo_with_context(args, topics, qrels, qid2contexts, bm25_searcher, use_context=True, ret_type='bm2', eval=True)
            eval_trec(all_qids, all_hits, args.eval_trec_mode, eval_prefix=f'iter{iter}')
            break
        # ----------------------------------------------------------------------------------------------------------

        print(f"{format_time()} Iter-{iter}: Retrieving demo_psgs with RM")
        all_hits = ret_demo_with_context(args, topics, qrels, qid2contexts, con_searcher, use_context=True, ret_type='de2', eval=False, query_encoder=query_encoder)
        # ----------------------------------------------------------------------------------------------------------

        demo_psgs = {}
        for qid, hits in zip(all_qids, all_hits):
            psgs = [psg_collections[int(hit.docid)][1].strip() for hit in hits[:args.num_demo]] 
            demo_psgs[qid] = '\n'.join([f"{str(k)}. {c}" for k,c in enumerate(psgs)])
        # ----------------------------------------------------------------------------------------------------------
