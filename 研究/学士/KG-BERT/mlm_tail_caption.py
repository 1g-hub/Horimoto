
import argparse
import logging
import numpy as np
from tqdm import tqdm

import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import torch
import tensorflow as tf
import pytorch_lightning as pl
from pytorch_lightning import Trainer

torch.use_deterministic_algorithms(True)
pl.seed_everything(42, workers=True)

from transformers import BertConfig, BertTokenizer, BertForMaskedLM, PreTrainedTokenizerBase
from transformers import DataCollatorForLanguageModeling, pipeline

logger = logging.getLogger(__name__)
    
parser = argparse.ArgumentParser()

def main():

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

    model_name = args.bert_model

    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
    bert_model = BertForMaskedLM(config)

    bert_model = bert_model.cuda()

    MASK_TOKEN = 103
    SEP_TOKEN = 102
    COMMA_TOKEN = 1010

    if args.do_train:

        train_triples = ["land reform, a redistribution of agricultural land (especially by government action)[SEP]hypernym[SEP]reform, a change for the better as a result of correcting abuses; \"justice was for sale before the reform of the law courts\"", 
                 "cover, provide ironman captain america hulk black widow hawkeye thor nick fuly wanda quicksilver ultron vision blackpanther with a covering or cause to be covered; \"cover her face with a handkerchief\"; \"cover the child with a blanket\"; \"cover the grave with flowers\"[SEP]derivationally related form[SEP]covering water, an artifact that covers something else (usually to protect or shelter or conceal it)"]

        val_triples = ["clangour, make a loud resonant noise; \"the alarm clangored throughout the building\"[SEP]hypernym[SEP]sound, make a certain noise or sound; \"She went `Mmmmm'\"; \"The gun went `bang'\""]

        test_triples = ["trade name, a name given to a product or service[SEP]member of domain usage[SEP][MASK], anticonvulsant drug (trade name Gemonil) used in the treatment of epilepsy"]
        
        # file_path = args.data_dir + '/train_triples_mask.txt'
        # with open(file_path) as f:
        #     train_triples = [s.rstrip() for s in f.readlines()]
        # print(train_triples[0:5])

        # input_ids・attention_maskをtokenizerから作成
        encodings = tokenizer(train_triples, return_tensors='pt', padding=True, truncation=True, max_length=args.max_seq_length)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # 各文に対してinput_ids・attention_mask・labelsのdictionaryを作成する
        col_all = []
        for i in range(len(attention_mask)):
            col = {}
            col['input_ids'] = input_ids[i].tolist()
            col['attention_mask'] = attention_mask[i].tolist()
            col['labels'] = input_ids[i].tolist()
            col_all.append(col)

        # [MASK]を自動生成する(15%の確率で[MASK]を生成)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.0)
        train_input_ids = data_collator(col_all)['input_ids']
        train_attention_mask = data_collator(col_all)['attention_mask']
        train_labels = data_collator(col_all)['labels']

        print("train_input_ids = ", train_input_ids)
        print("train_labels = ", train_labels)


        # print("data_collator = ", data_collator)
        # print("train_input_ids = ", train_input_ids[0:5])
        # print("train_attention_mask = ", train_attention_mask[0:5])
        # print("train_labels = ", train_labels[0:5])

        for num_inputs, _inputs in enumerate(train_input_ids):
            sep_count = 0
            for index, _input in enumerate(_inputs):
                if _input == SEP_TOKEN:
                    # print("_input = ", _input)
                    sep_count += 1
                    if sep_count == 2:
                        for word_count in range(10):
                            # print("train_input_ids[num_inputs][index+word_count+1] = ", train_input_ids[num_inputs][index+word_count+1])
                            train_labels[num_inputs][index+word_count+1] = train_input_ids[num_inputs][index+word_count+1]
                            train_input_ids[num_inputs][index+word_count+1] = MASK_TOKEN
                            if train_input_ids[num_inputs][index+word_count+2] == COMMA_TOKEN:
                                break

        # print("MASK")
        # print("train_input_ids = ", train_input_ids[0:5])
        # print("train_attention_mask = ", train_attention_mask[0:5])
        # print("train_labels = ", train_labels[0:5])

        # file_path = args.data_dir + '/dev_triples_mask.txt'
        # with open(file_path) as f:
        #     val_triples = [s.rstrip() for s in f.readlines()]
        # print(val_triples[0:5])

        # input_ids・attention_maskをtokenizerから作成
        encodings = tokenizer(val_triples, return_tensors='pt', padding=True, truncation=True, max_length=args.max_seq_length)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # 各文に対してinput_ids・attention_mask・labelsのdictionaryを作成する
        col_all = []
        for i in range(len(attention_mask)):
            col = {}
            col['input_ids'] = input_ids[i].tolist()
            col['attention_mask'] = attention_mask[i].tolist()
            col['labels'] = input_ids[i].tolist()
            col_all.append(col)

        # [MASK]を自動生成する(15%の確率で[MASK]を生成)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.0)
        val_input_ids = data_collator(col_all)['input_ids']
        val_attention_mask = data_collator(col_all)['attention_mask']
        val_labels = data_collator(col_all)['labels']

        print("val_input_ids = ", val_input_ids)
        print("val_labels = ", val_labels)

        # print("data_collator = ", data_collator)
        # print("val_input_ids = ", val_input_ids[0:5])
        # print("val_attention_mask = ", val_attention_mask[0:5])
        # print("val_labels = ", val_labels[0:5])

        for num_inputs, _inputs in enumerate(val_input_ids):
            sep_count = 0
            for index, _input in enumerate(_inputs):
                if _input == SEP_TOKEN:
                    # print("_input = ", _input)
                    sep_count += 1
                    if sep_count == 2:
                        for word_count in range(10):
                            # print("val_input_ids[num_inputs][index+word_count+1] = ", val_input_ids[num_inputs][index+word_count+1])
                            val_labels[num_inputs][index+word_count+1] = val_input_ids[num_inputs][index+word_count+1]
                            val_input_ids[num_inputs][index+word_count+1] = MASK_TOKEN
                            if val_input_ids[num_inputs][index+word_count+2] == COMMA_TOKEN:
                                break

        # print("MASK")
        # print("val_input_ids = ", val_input_ids[0:5])
        # print("val_attention_mask = ", val_attention_mask[0:5])
        # print("val_labels = ", val_labels[0:5])

        # file_path = args.data_dir + '/test_triples_mask.txt'
        # with open(file_path) as f:
        #     test_triples = [s.rstrip() for s in f.readlines()]
        # print(test_triples[0:5])

        # input_ids・attention_maskをtokenizerから作成
        encodings = tokenizer(test_triples, return_tensors='pt', padding=True, truncation=True, max_length=args.max_seq_length)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # 各文に対してinput_ids・attention_mask・labelsのdictionaryを作成する
        col_all = []
        for i in range(len(attention_mask)):
            col = {}
            col['input_ids'] = input_ids[i].tolist()
            col['attention_mask'] = attention_mask[i].tolist()
            col['labels'] = input_ids[i].tolist()
            col_all.append(col)

        # [MASK]を自動生成する(15%の確率で[MASK]を生成)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.0)
        test_input_ids = data_collator(col_all)['input_ids']
        test_attention_mask = data_collator(col_all)['attention_mask']
        test_labels = data_collator(col_all)['labels']

        print("test_input_ids = ", test_input_ids)
        print("test_labels = ", test_labels)

        # print("data_collator = ", data_collator)
        # print("test_input_ids = ", test_input_ids[0:5])
        # print("test_attention_mask = ", test_attention_mask[0:5])
        # print("test_labels = ", test_labels[0:5])

        for num_inputs, _inputs in enumerate(test_input_ids):
            sep_count = 0
            for index, _input in enumerate(_inputs):
                if _input == SEP_TOKEN:
                    # print("_input = ", _input)
                    sep_count += 1
                    if sep_count == 2:
                        for word_count in range(10):
                            # print("test_input_ids[num_inputs][index+word_count+1] = ", test_input_ids[num_inputs][index+word_count+1])
                            test_labels[num_inputs][index+word_count+1] = test_input_ids[num_inputs][index+word_count+1]
                            test_input_ids[num_inputs][index+word_count+1] = MASK_TOKEN
                            if test_input_ids[num_inputs][index+word_count+2] == COMMA_TOKEN:
                                break

        # print("MASK")
        # print("test_input_ids = ", test_input_ids[0:5])
        # print("test_attention_mask = ", test_attention_mask[0:5])
        # print("test_labels = ", test_labels[0:5])

        # Dataset定義
        class Datasets(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                input = self.input_ids[idx]
                mask = self.attention_mask[idx]
                label = self.labels[idx]
                return input, mask, label

        # DataModule定義
        class DataModule(pl.LightningDataModule):
            def __init__(self, train_input_ids, train_attention_mask, train_labels, val_input_ids, val_attention_mask, val_labels, test_input_ids, test_attention_mask, test_labels, batch_size, num_workers):
                super().__init__()
                self.train_input_ids = train_input_ids
                self.train_attention_mask = train_attention_mask
                self.train_labels = train_labels
                self.val_input_ids = val_input_ids
                self.val_attention_mask = val_attention_mask
                self.val_labels = val_labels
                self.test_input_ids = test_input_ids
                self.test_attention_mask = test_attention_mask
                self.test_labels = test_labels
                self.save_hyperparameters() 
            # 訓練・検証時のデータセット作成
            def prepare_data(self):
                self.train_dataset = Datasets(self.train_input_ids, self.train_attention_mask, self.train_labels)
                self.val_dataset = Datasets(self.val_input_ids, self.val_attention_mask, self.val_labels)
                self.test_dataset = Datasets(self.test_input_ids, self.test_attention_mask, self.test_labels)
            # 訓練時のデータローダ作成
            def train_dataloader(self):
                dataloader = torch.utils.data.DataLoader(
                    self.train_dataset, batch_size=self.hparams.batch_size
                )
                return dataloader
            # 検証時のデータローダ作成
            def val_dataloader(self):
                dataloader = torch.utils.data.DataLoader(
                    self.val_dataset, batch_size=self.hparams.batch_size
                )
                return dataloader
            # テスト時のデータローダ作成
            def test_dataloader(self):
                dataloader = torch.utils.data.DataLoader(
                    self.test_dataset, batch_size=self.hparams.batch_size
                )
                return dataloader

        # ModelModule定義
        class ModelModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model_name = "bert-base-uncased"
                self.model = BertForMaskedLM.from_pretrained(self.model_name)
            # 順伝播
            def forward(self, x):
                x = self.model(x)
                return x
            # 損失関数・スケジューラ定義
            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
                return [optimizer], [lr_scheduler]
            # 訓練時
            def training_step(self, batch, batch_idx):
                input, mask, label = batch
                loss = self.model(input_ids=input, labels=label).loss
                self.log('train_loss', loss)
                return loss
            # 検証時
            def validation_step(self, batch, batch_idx):
                input, mask, label = batch
                loss = self.model(input_ids=input, labels=label).loss
                self.log('val_loss', loss)
                return loss
            # テスト時
            def test_dataloader(self):
                return torch.utils.data.DataLoader(self.test_dataset, self.batch_size)
            def test_step(self, batch, batch_idx):
                input, mask, label = batch
                loss = self.model(input_ids=input, labels=label).loss
                acc = torch.sum(input == label) * 1.0 / len(input)
                results = {'test_loss': loss, 'test_acc': acc}
                return results

            def test_end(self, outputs):
                avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
                avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
                results = {'test_loss': avg_loss, 'test_acc': avg_acc}
                return results

        # モデルの重みを保存する条件を指定
        save_point = pl.callbacks.ModelCheckpoint(
            monitor = 'val_loss', 
            mode = 'min',
            save_top_k = 1, # 保存するモデルの数
            save_weights_only=True,
            dirpath = args.output_dir # モデルの保存先
        )

        # Early Stopping
        early_stopping = pl.callbacks.EarlyStopping(
            monitor = 'val_loss',
            mode = 'min',
            patience = 5
        )

        # 学習の大枠を定義（今回は二つのCallbacksを利用）
        trainer = pl.Trainer(
            max_epochs= 5, #args.num_train_epochs,
            # gpus = -1 if torch.cuda.is_available() else None,
            callbacks = [save_point, early_stopping],
            deterministic=True
        )

        data = DataModule(train_input_ids, train_attention_mask, train_labels, val_input_ids, val_attention_mask, val_labels, test_input_ids, test_attention_mask, test_labels, batch_size=args.train_batch_size, num_workers=8)
        model = ModelModule().to(device)

        print(model)

        # 訓練・検証
        print("\nTraining Start : \n")
        trainer.fit(model, data)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):



        # trainer.test()

        model_predict = ModelModule.load_from_checkpoint(checkpoint_path=save_point.best_model_path)
        # model_predict = model_predict.cuda()
        # model_predict.eval()

        # print(model_predict)

        text = 'trade name, a name given to a product or service[SEP]member of domain usage[SEP][MASK], anticonvulsant drug (trade name Gemonil) used in the treatment of epilepsy'
        answer = "metharbital"
        # tokens = tokenizer.tokenize(text)

        # file_path = args.data_dir + '/test_triples_mask.txt'
        # with open(file_path) as f:
        #     test_triples = [s.rstrip() for s in f.readlines()]
        # print(test_triples[0:5])

        # result_file = os.path.join(args.output_dir, "predict_tail_test1.txt")

        # n = 0
        # rank = []
        # answer = []
        # mask_test_triples = []
        # num_test_triples = len(test_triples)
        # for num_triples, tmp_test_triple in enumerate(tqdm(test_triples)):
        #     n += 1
        #     sep_index = [i for i, x in enumerate(tmp_test_triple) if x == ']']
        #     tail_index = tmp_test_triple[sep_index[1]:].index(",")

        #     _answer = tmp_test_triple[sep_index[1]+1:sep_index[1]+tail_index].split()
        #     answer.append(_answer)
        #     mask_count = len(_answer)
        #     # if n < 10:
        #     #     print("tmp_test_triple = ", tmp_test_triple)
        #     #     print("sep_index = ", sep_index)
        #     #     print("tail_index = ", tail_index)
        #     #     print("mask_count = ", mask_count)

        #     mask_triple = tmp_test_triple[:sep_index[1]+1] + MASK_TOKEN*mask_count # tmp_test_triple[sep_index[1]+tail_index:]
        #     mask_test_triples.append(mask_triple)

        #     input_ids = tokenizer.encode(mask_triple, return_tensors='pt')
        #     input_ids = input_ids.cuda()

        #     with torch.no_grad():
        #         output = model_predict(input_ids=input_ids)
        #         scores = output.logits

        #     mask_position = input_ids[0].tolist().index(4)

        #     with open(result_file, mode='w') as f:
        #         f.write(f"\n{num_triples} -> {tmp_test_triple}\n")
        #         f.write(f"\tanswer = {_answer}\n")
        #         for k in range(10):
        #             id_best = scores[k, mask_position].argmax(-1).item()
        #             token_best  = tokenizer.convert_ids_to_tokens(id_best)
        #             # token_best = token_best.replace('##', '')
        #             f.write(f"\t\ttop_{k} = {token_best}\n")

        #             if token_best == _answer:
        #                 rank.append(k)
        #             elif token_best != _answer and k == 9:
        #                 rank.append(99)

        # print("mask_test_triples = ", mask_test_triples[0:5])
        # print(answer[0:5])

        # hits_k = [0]*3
        # for _rank in rank:
        #     if _rank == 0:
        #         hits_k[0] += 1
        #     if _rank < 3:
        #         hits_k[1] += 1
        #     if _rank < 10:
        #         hits_k[2] += 1

        # print("Hits@1 = ", hits_k[0]/num_test_triples)
        # print("Hits@3 = ", hits_k[1]/num_test_triples)
        # print("Hits@10 = ", hits_k[2]/num_test_triples)


        input_ids = tokenizer.encode(text, return_tensors='pt')
        # input_ids = input_ids.cuda()


        # print("tokens = ", ["[CLS]"]+tokens+["[SEP]"])
        print("input_ids = ", input_ids)

        with torch.no_grad():
            output = model_predict(input_ids=input_ids)
            scores = output.logits

        mask_position = input_ids[0].tolist().index(103)
        id_best = scores[0, mask_position].argmax(-1).item()
        token_best  = tokenizer.convert_ids_to_tokens(id_best)
        token_best = token_best.replace('##', '')
        print("token_best = ", token_best)

        text = text.replace('[MASK]', token_best)
        print(text)

if __name__ == "__main__":
    main()
