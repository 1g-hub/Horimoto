

import argparse
import csv
import sys
import numpy as np
from tqdm import tqdm, trange

import json
import os
import pickle
import random
import time
import warnings
import logging
from typing import Dict, List, Optional

import torch
from filelock import FileLock
from torch.utils.data import Dataset

from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, pipeline
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    # parser.add_argument("--task_name",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--mlm_probability",
                        default=0.15,
                        type=float,
                        help="The probability of Mask when training. ")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

        

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # text = "[CLS] My room is White. [SEP]"
    # print(tokenizer.tokenize(text))

    config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
    model = BertForMaskedLM(config)

    model_save_path = args.output_dir

    print(model)

    if args.do_train:

        model.to(device)
        model.train()

        dataset = LineByLineTextDataset(
            tokenizer = tokenizer,
            file_path = args.data_dir + '/train_dev_triples.txt',
            block_size = args.max_seq_length, # tokenizerのmax_length
        )

        # print("dataset = ", dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True,
            mlm_probability= args.mlm_probability
        )

        # print("data_collator = ", data_collator)

        training_args = TrainingArguments(
            output_dir= args.output_dir, # output_dir = 'mlm_result/output_1'
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            save_steps=10000,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=args.learning_rate,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )

        print("trainer = ", trainer.train_dataset)

        trainer.train()
        trainer.save_model(model_save_path)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained(model_save_path)

        model.eval()

        fill_mask = pipeline(
            "fill-mask",
            model=model,
            tokenizer=tokenizer,
            top_k=100
        )

        MASK_TOKEN = tokenizer.mask_token
        # print(MASK_TOKEN)

        file_path = args.data_dir + '/test_triples.txt'
        with open(file_path) as f:
            test_triples = [s.rstrip() for s in f.readlines()]
        # print(test_triples[0:5])

        n = 0
        answer = []
        mask_test_triples = []
        num_test_triples = len(test_triples)
        test_head_rel = []
        for tmp_test_triple in test_triples:
            n += 1
            sep_index = [i for i, x in enumerate(tmp_test_triple) if x == ']']
            tail_index = tmp_test_triple[sep_index[1]:].index(",")

            test_head_rel.append(tmp_test_triple[:sep_index[1]+1])
            _answer = tmp_test_triple[sep_index[1]+1:sep_index[1]+tail_index].split()
            answer.append(_answer)
            mask_count = len(_answer)
            # if n < 10:
            #     print("tmp_test_triple = ", tmp_test_triple)
            #     print("sep_index = ", sep_index)
            #     print("tail_index = ", tail_index)
            #     print("mask_count = ", mask_count)

            mask_triple = tmp_test_triple[:sep_index[1]+1] + MASK_TOKEN*mask_count + "[SEP]" # tmp_test_triple[sep_index[1]+tail_index:]
            mask_test_triples.append(mask_triple)
        # print("mask_test_triples = ", mask_test_triples[0:10])
        # print("test_head_rel, answer = ", test_head_rel[0:10], answer[0:10])

        result = []
        for triples in tqdm(mask_test_triples):
            result.append(fill_mask(triples))
        
        # for _result in result[0:5]:
        #     print(_result)

        file_path = args.data_dir + '/all_triples.txt'
        with open(file_path) as f:
            all_triples = [s.rstrip() for s in f.readlines()]
        # print("all_triples = ", all_triples[0:5])

        n = 0
        head_rel, tail_caption = [], []
        for tmp_triple in all_triples:
            n += 1
            sep_index = [i for i, x in enumerate(tmp_triple) if x == ']']
            tail_index = tmp_triple[sep_index[1]:].index(",")

            head_rel.append(tmp_triple[:sep_index[1]+1])
            tail_caption.append(tmp_triple[sep_index[1]+1:sep_index[1]+tail_index])
            
            # if n < 5:
            #     print("head_rel = ", head_rel)
            #     print("tail_caption = ", tail_caption)
            #     print("sep_index = ", sep_index)
            #     print("tail_index = ", tail_index)
        
        dic_triples = {}
        for h_r, t_cap in zip(head_rel, tail_caption):
            dic_triples.setdefault(h_r, []).append(t_cap)

        rank = []
        result_file = os.path.join(args.output_dir, "predict_tail.txt")
        with open(result_file, mode='w') as f:
            for num_triples in range(len(mask_test_triples)):
                if len(result[num_triples]) == 100:
                    f.write(f"\n{num_triples} -> answer = {answer[num_triples][0]}\n")
                    n = 0
                    for top_k, _result in enumerate(result[num_triples]):
                        if _result['token_str'] == answer[num_triples][0]:
                            n = 1
                            _rank = top_k
                            rank.append(_rank)
                        f.write(f"  top_k, result = {top_k}, {_result['token_str']}\n")
                    if n == 0:
                        rank.append(40943)
                else:
                    _rank = []
                    f.write(f"\nNumber of tail words = {len(result[num_triples])}\n")
                    for num_mask in range(len(result[num_triples])):
                        f.write(f"{num_triples} -> answer = {answer[num_triples][num_mask]}\n")
                        for top_k, _result in enumerate(result[num_triples][num_mask]):
                            if _result['token_str'] == answer[num_triples][num_mask]:
                                _rank.append(top_k)
                            f.write(f"  top_k result = {top_k} {_result['token_str']} \n")
                    if len(_rank) != len(result[num_triples]):
                        for c in range(len(result[num_triples]) - len(_rank)):
                            _rank.append(40943)
                    rank.append(sum(_rank)/len(_rank))

        hits_k = [0]*3
        for _rank in rank:
            if _rank == 0:
                hits_k[0] += 1
            if _rank < 3:
                hits_k[1] += 1
            if _rank < 10:
                hits_k[2] += 1

        rank_1 = [_rank + 1  for _rank in rank]
        
        sum_reci_rank = 0
        for _rank in rank_1:
            sum_reci_rank += 1/ _rank
        
        print("\nHits@1 = ", hits_k[0]/num_test_triples)
        print("Hits@3 = ", hits_k[1]/num_test_triples)
        print("Hits@10 = ", hits_k[2]/num_test_triples)
        print("Mean Rank = ", sum(rank_1)/len(rank_1))
        print("Mean Reciprocal Rank = ", sum_reci_rank/len(rank))

        rank_file = os.path.join(args.output_dir, "result.txt")
        with open(rank_file, mode='a') as f:
            f.write(f"\nHits@1  = {hits_k[0]/num_test_triples} \n")
            f.write(f"Hits@3  = {hits_k[1]/num_test_triples} \n")
            f.write(f"Hits@10 = {hits_k[2]/num_test_triples} \n")
            f.write(f"Mean Rank = {sum(rank_1)/len(rank_1)} \n")
            f.write(f"Mean Reciprocal Rank = {sum_reci_rank/len(rank)} \n")
 

        for test_i, (test_h_r, ans, _result) in enumerate(zip(test_head_rel, answer, result)):
            # print("rank = ", rank[test_i])
            for tail_cap in dic_triples[test_h_r]:
                if len(_result) == 100:
                    for i, __result in enumerate(_result):
                        # print("i, __result = ", i, __result['token_str'])
                        if __result['token_str'] == ans[0]:
                            # print("i, ans, rank = ", i, ans[0], rank[test_i])
                            break
                        if __result['token_str'] == tail_cap:
                            # print("i, tail_cap = ", i, tail_cap)
                            rank[test_i] -= 1
                else:
                    for num_mask in range(len(_result)):
                        for i, __result in enumerate(_result[num_mask]):
                            # print("i, __result = ", i, __result['token_str'])
                            if __result['token_str'] == ans[num_mask]:
                                # print("i, ans, rank = ", i, ans[num_mask], rank[test_i])
                                break
                            if len(_result) == len(tail_cap.split()) and __result['token_str'] == tail_cap.split()[num_mask]:
                                # print("i, num_mask, tail_cap = ", i, num_mask, tail_cap)
                                rank[test_i] -= 1


        # print(len(rank), len(test_triples))
        # print("rank = ", rank)
        rank_file = os.path.join(args.output_dir, "rank_triples.txt")
        with open(rank_file, mode='w') as f:
            for i, (_rank, _triples) in enumerate(zip(rank, test_triples)):
                f.write(f"{i} -> {_rank}  {_triples} \n")
        
        hits_k = [0]*3
        for _rank in rank:
            if _rank == 0:
                hits_k[0] += 1
            if _rank < 3:
                hits_k[1] += 1
            if _rank < 10:
                hits_k[2] += 1
        
        rank_1 = [_rank + 1  for _rank in rank]
        
        sum_reci_rank = 0
        for _rank in rank_1:
            if _rank <= 0:
                _rank = 1
            sum_reci_rank += 1/ _rank
        
        print("\nFillter Hits@1 = ", hits_k[0]/num_test_triples)
        print("Fillter Hits@3 = ", hits_k[1]/num_test_triples)
        print("Fillter Hits@10 = ", hits_k[2]/num_test_triples)
        print("Fillter Mean Rank = ", sum(rank_1)/len(rank_1))
        print("Fillter Mean Reciprocal Rank = ", sum_reci_rank/len(rank))

        rank_file = os.path.join(args.output_dir, "result.txt")
        with open(rank_file, mode='a') as f:
            f.write(f"\nFillter Hits@1  = {hits_k[0]/num_test_triples} \n")
            f.write(f"Fillter Hits@3  = {hits_k[1]/num_test_triples} \n")
            f.write(f"Fillter Hits@10 = {hits_k[2]/num_test_triples} \n")
            f.write(f"Fillter Mean Rank = {sum(rank_1)/len(rank_1)} \n")
            f.write(f"Fillter Mean Reciprocal Rank = {sum_reci_rank/len(rank)} \n")
 
if __name__ == "__main__":
    main()
