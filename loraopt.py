import torch
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, AutoModel, PretrainedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler, PreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.llama.modeling_llama import *
from transformers.models.opt.modeling_opt import *
from transformers.models.bloom.modeling_bloom import *
from transformers.tokenization_utils_base import *
from transformers.data.data_collator import *
from dataclasses import dataclass
import json
from torch.utils import *
from torch.utils.data import DataLoader
from torch.nn import *
import argparse
import csv
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, Accelerator
import deepspeed
from datasets import Dataset, load_dataset, concatenate_datasets
import copy as cp
import numpy as np
from deepspeed.runtime.utils import see_memory_usage
from torch.nn.utils.rnn import pad_sequence
import math
from typing import List, Optional, Tuple, Union
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
import torch.nn.functional as F
from tqdm import tqdm
from transformers.utils import is_torch_tpu_available
from baukit import Trace, TraceDict



parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,help='模型名称')
parser.add_argument('--dataset_name', type=str,help='数据集名称')
parser.add_argument('--mode', type=str,help='训练策略')
parser.add_argument('--cur_epoch', type=int,help='训练策略')
parser.add_argument('--worst_k', type=int,help='训练策略')
parser.add_argument('--domain_name', type=str,help='数据集域名')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
parser.add_argument("--generation_max_length", type=int, default=2000, help="Maximum length to use for generation")
parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate to use for training.")
parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
parser.add_argument(
    "--bf16",
    type=bool,
    default=True if torch.cuda.get_device_capability()[0] == 8 else False,
    help="Whether to use bf16.",
)
args = parser.parse_args()
print(args.dataset_name)
print(args.model_name)
label_names = None
if args.mode == 't_y_f_y':
    label_names = ["labels"]
if args.mode == 't_y_f_n':
    label_names = ["labels","true_false_labels"]
if args.mode == 'con':
    label_names = ["labels","true_false_labels"]
if args.mode == 'procon' or args.mode == 'proun':
    label_names = ["labels","true_false_labels"]


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
IGNORE_INDEX = -100
index_list = [1, 2, 3, 0]
index_list = index_list[:args.worst_k]
k_proj = []
v_proj = []
q_proj = []
o_proj = []
fc1 = []
fc2 = []
for i in index_list:
    k_proj.append(f"model.decoder.layers.{i}.self_attn.k_proj.weight")
    v_proj.append(f"model.decoder.layers.{i}.self_attn.v_proj.weight")
    q_proj.append(f"model.decoder.layers.{i}.self_attn.q_proj.weight")
    o_proj.append(f"model.decoder.layers.{i}.self_attn.out_proj.weight")
    fc1.append(f"model.decoder.layers.{i}.fc1.weight")
    fc2.append(f"model.decoder.layers.{i}.fc2.weight")
modules_list = k_proj + v_proj + q_proj + o_proj + fc1 + fc2
LAYER_NUM = 28


tokenizer = AutoTokenizer.from_pretrained("../model/"+args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

def batch_preprocessing_t_y_f_y(batch):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    if args.dataset_name == 'XSUM':
        for article, summary, sen_label in zip(batch['article'],batch['summary'],batch['label']):
            if len(sen_label) == 0:
                prompt = "Article: " + article + "\nWrite a summary consistent with the above article in no more than 40 words:\n"
            else:
                prompt = "Article: " + article + "\nWrite a summary inconsistent with the above article in no more than 40 words:\n"
                #(1,seq_len)
            prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_tokenized['labels'] = torch.full_like(prompt_tokenized['input_ids'], IGNORE_INDEX)
            if len(sen_label) == 0:#真实
                for summary_sen_index in range(len(summary)):
                    summary_sen_ids = tokenizer(summary[summary_sen_index], return_tensors="pt", add_special_tokens=False)
                    #(1,seq_len)
                    prompt_tokenized['input_ids'] = torch.cat((prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                    prompt_tokenized['attention_mask'] = torch.cat((prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                    prompt_tokenized['labels'] = torch.cat((prompt_tokenized['labels'], summary_sen_ids['input_ids']),dim=1)
            else:#幻觉
                for summary_sen_index in range(len(summary)):
                    summary_sen_ids = tokenizer(summary[summary_sen_index], return_tensors="pt", add_special_tokens=False)
                    if summary_sen_index in sen_label:
                        prompt_tokenized['input_ids'] = torch.cat((prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                        prompt_tokenized['attention_mask'] = torch.cat((prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                        prompt_tokenized['labels'] = torch.cat((prompt_tokenized['labels'], summary_sen_ids['input_ids']),dim=1)
                    else:
                        prompt_tokenized['input_ids'] = torch.cat((prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                        prompt_tokenized['attention_mask'] = torch.cat((prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                        prompt_tokenized['labels'] = torch.cat((prompt_tokenized['labels'], torch.full_like(summary_sen_ids['input_ids'], IGNORE_INDEX)),dim=1)
            eos_token_tokenized = tokenizer(tokenizer.eos_token, return_tensors="pt", add_special_tokens=False)
            prompt_tokenized['input_ids'] = torch.cat((prompt_tokenized['input_ids'], eos_token_tokenized['input_ids']),dim=1)
            prompt_tokenized['attention_mask'] = torch.cat((prompt_tokenized['attention_mask'], eos_token_tokenized['attention_mask']),dim=1)
            prompt_tokenized['labels'] = torch.cat((prompt_tokenized['labels'], eos_token_tokenized['input_ids']),dim=1)
            batch_input_ids.append(prompt_tokenized['input_ids'].squeeze())
            batch_attention_mask.append(prompt_tokenized['attention_mask'].squeeze())
            batch_labels.append(prompt_tokenized['labels'].squeeze())
        example = {}
        example['input_ids'] = pad_sequence(batch_input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)
        example['attention_mask'] = pad_sequence(batch_attention_mask,batch_first=True,padding_value=0)
        example['labels'] = pad_sequence(batch_labels,batch_first=True,padding_value=-100)
        if example['input_ids'].shape[1] > 2000:
            example['input_ids'] = example['input_ids'][:,:2000]
            example['attention_mask'] = example['attention_mask'][:,:2000]
            example['labels'] = example['labels'][:,:2000]
        return example
    if args.dataset_name == 'CNNDM':
        return
    if args.dataset_name == 'TWEET':
        return
    if args.dataset_name == 'MEDIA':
        return
def batch_preprocessing_t_y_f_n(batch):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_tf_labels = []
    if args.dataset_name == 'XSUM':
        for article, summary, sen_label in zip(batch['article'],batch['summary'],batch['label']):
            prompt = "Article: " + article + "\nWrite a summary consistent with the above article in no more than 40 words:\n"
            prompt_tokenized = tokenizer(prompt, return_tensors="pt")
            prompt_tokenized['labels'] = torch.full_like(prompt_tokenized['input_ids'], IGNORE_INDEX)
            if len(sen_label) == 0:
                batch_tf_labels.append(1)
                for summary_sen_index in range(len(summary)):
                    summary_sen_ids = tokenizer(summary[summary_sen_index], return_tensors="pt", add_special_tokens=False)
                    #(1,seq_len)
                    prompt_tokenized['input_ids'] = torch.cat((prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                    prompt_tokenized['attention_mask'] = torch.cat((prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                    prompt_tokenized['labels'] = torch.cat((prompt_tokenized['labels'], summary_sen_ids['input_ids']),dim=1)
            else:
                batch_tf_labels.append(0)
                for summary_sen_index in range(len(summary)):
                    summary_sen_ids = tokenizer(summary[summary_sen_index], return_tensors="pt", add_special_tokens=False)
                    if summary_sen_index in sen_label:
                        prompt_tokenized['input_ids'] = torch.cat((prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                        prompt_tokenized['attention_mask'] = torch.cat((prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                        prompt_tokenized['labels'] = torch.cat((prompt_tokenized['labels'], summary_sen_ids['input_ids']),dim=1)
                    else:
                        prompt_tokenized['input_ids'] = torch.cat((prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                        prompt_tokenized['attention_mask'] = torch.cat((prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                        prompt_tokenized['labels'] = torch.cat((prompt_tokenized['labels'], torch.full_like(summary_sen_ids['input_ids'], IGNORE_INDEX)),dim=1)
            eos_token_tokenized = tokenizer(tokenizer.eos_token, return_tensors="pt", add_special_tokens=False)
            prompt_tokenized['input_ids'] = torch.cat((prompt_tokenized['input_ids'], eos_token_tokenized['input_ids']),dim=1)
            prompt_tokenized['attention_mask'] = torch.cat((prompt_tokenized['attention_mask'], eos_token_tokenized['attention_mask']),dim=1)
            prompt_tokenized['labels'] = torch.cat((prompt_tokenized['labels'], eos_token_tokenized['input_ids']),dim=1)
            batch_input_ids.append(prompt_tokenized['input_ids'].squeeze())
            batch_attention_mask.append(prompt_tokenized['attention_mask'].squeeze())
            batch_labels.append(prompt_tokenized['labels'].squeeze())
        example = {}
        example['input_ids'] = pad_sequence(batch_input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)
        example['attention_mask'] = pad_sequence(batch_attention_mask,batch_first=True,padding_value=0)
        example['labels'] = pad_sequence(batch_labels,batch_first=True,padding_value=-100)
        if example['input_ids'].shape[1] > 2000:
            example['input_ids'] = example['input_ids'][:,:2000]
            example['attention_mask'] = example['attention_mask'][:,:2000]
            example['labels'] = example['labels'][:,:2000]
        example['true_false_labels'] = torch.tensor(batch_tf_labels)
        return example
    if args.dataset_name == 'CNNDM':
        return
    if args.dataset_name == 'TWEET':
        return
    if args.dataset_name == 'MEDIA':
        return
    
def batch_preprocessing_con(batch):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_con_labels = []
    if args.dataset_name == 'XSUM':
        for article, summary, sen_label in zip(batch['article'],batch['summary'],batch['label']):
            if len(sen_label) == 0:
                consist_prompt = "Article: " + article + "\nWrite a summary consistent with the above article in no more than 40 words:\n"
                conflict_prompt = "Article: " + article + "\nWrite a summary inconsistent with the above article in no more than 40 words:\n"
            else:
                consist_prompt = "Article: " + article + "\nWrite a summary inconsistent with the above article in no more than 40 words:\n"
                conflict_prompt = "Article: " + article + "\nWrite a summary consistent with the above article in no more than 40 words:\n"
            #(1,seq_len)
            consist_prompt_tokenized = tokenizer(consist_prompt, return_tensors="pt")
            consist_prompt_tokenized['labels'] = torch.full_like(consist_prompt_tokenized['input_ids'], IGNORE_INDEX)
            conflict_prompt_tokenized = tokenizer(conflict_prompt, return_tensors="pt")
            conflict_prompt_tokenized['labels'] = torch.full_like(conflict_prompt_tokenized['input_ids'], IGNORE_INDEX)

            if len(sen_label) == 0:#真实
                for summary_sen_index in range(len(summary)):
                    summary_sen_ids = tokenizer(summary[summary_sen_index], return_tensors="pt", add_special_tokens=False)
                    #(1,seq_len)
                    consist_prompt_tokenized['input_ids'] = torch.cat((consist_prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                    consist_prompt_tokenized['attention_mask'] = torch.cat((consist_prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                    consist_prompt_tokenized['labels'] = torch.cat((consist_prompt_tokenized['labels'], summary_sen_ids['input_ids']),dim=1)
                    #conflict指令中，所有无幻觉的句子都要被惩罚
                    conflict_prompt_tokenized['input_ids'] = torch.cat((conflict_prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                    conflict_prompt_tokenized['attention_mask'] = torch.cat((conflict_prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                    conflict_prompt_tokenized['labels'] = torch.cat((conflict_prompt_tokenized['labels'], summary_sen_ids['input_ids']),dim=1)
            else:#幻觉
                for summary_sen_index in range(len(summary)):
                    summary_sen_ids = tokenizer(summary[summary_sen_index], return_tensors="pt", add_special_tokens=False)
                    if summary_sen_index in sen_label:
                        #consist指令中，有幻觉的句子直接拼，符合指令，cross entropy鼓励
                        consist_prompt_tokenized['input_ids'] = torch.cat((consist_prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                        consist_prompt_tokenized['attention_mask'] = torch.cat((consist_prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                        consist_prompt_tokenized['labels'] = torch.cat((consist_prompt_tokenized['labels'], summary_sen_ids['input_ids']),dim=1)
                        #conflict指令中，有幻觉的句子直接拼，有冲突，都要被惩罚
                        conflict_prompt_tokenized['input_ids'] = torch.cat((conflict_prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                        conflict_prompt_tokenized['attention_mask'] = torch.cat((conflict_prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                        conflict_prompt_tokenized['labels'] = torch.cat((conflict_prompt_tokenized['labels'], summary_sen_ids['input_ids']),dim=1)
                    else:
                        #consist中，无幻觉的句子mask，没有符合指令，不用鼓励
                        consist_prompt_tokenized['input_ids'] = torch.cat((consist_prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                        consist_prompt_tokenized['attention_mask'] = torch.cat((consist_prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                        consist_prompt_tokenized['labels'] = torch.cat((consist_prompt_tokenized['labels'], torch.full_like(summary_sen_ids['input_ids'], IGNORE_INDEX)),dim=1)
                        #conflict中，无幻觉的句子mask，没有冲突，不用惩罚
                        conflict_prompt_tokenized['input_ids'] = torch.cat((conflict_prompt_tokenized['input_ids'], summary_sen_ids['input_ids']),dim=1)
                        conflict_prompt_tokenized['attention_mask'] = torch.cat((conflict_prompt_tokenized['attention_mask'], summary_sen_ids['attention_mask']),dim=1)
                        conflict_prompt_tokenized['labels'] = torch.cat((conflict_prompt_tokenized['labels'], torch.full_like(summary_sen_ids['input_ids'], IGNORE_INDEX)),dim=1)
            eos_token_tokenized = tokenizer(tokenizer.eos_token, return_tensors="pt", add_special_tokens=False)
            consist_prompt_tokenized['input_ids'] = torch.cat((consist_prompt_tokenized['input_ids'], eos_token_tokenized['input_ids']),dim=1)
            consist_prompt_tokenized['attention_mask'] = torch.cat((consist_prompt_tokenized['attention_mask'], eos_token_tokenized['attention_mask']),dim=1)
            consist_prompt_tokenized['labels'] = torch.cat((consist_prompt_tokenized['labels'], eos_token_tokenized['input_ids']),dim=1)
            conflict_prompt_tokenized['input_ids'] = torch.cat((conflict_prompt_tokenized['input_ids'], eos_token_tokenized['input_ids']),dim=1)
            conflict_prompt_tokenized['attention_mask'] = torch.cat((conflict_prompt_tokenized['attention_mask'], eos_token_tokenized['attention_mask']),dim=1)
            conflict_prompt_tokenized['labels'] = torch.cat((conflict_prompt_tokenized['labels'], eos_token_tokenized['input_ids']),dim=1)
            batch_input_ids.append(consist_prompt_tokenized['input_ids'].squeeze())
            batch_attention_mask.append(consist_prompt_tokenized['attention_mask'].squeeze())
            batch_labels.append(consist_prompt_tokenized['labels'].squeeze())
            batch_con_labels.append(1)
            batch_input_ids.append(conflict_prompt_tokenized['input_ids'].squeeze())
            batch_attention_mask.append(conflict_prompt_tokenized['attention_mask'].squeeze())
            batch_labels.append(conflict_prompt_tokenized['labels'].squeeze())
            batch_con_labels.append(0)
        example = {}
        example['input_ids'] = pad_sequence(batch_input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)
        example['attention_mask'] = pad_sequence(batch_attention_mask,batch_first=True,padding_value=0)
        example['labels'] = pad_sequence(batch_labels,batch_first=True,padding_value=-100)
        if example['input_ids'].shape[1] > 2000:
            example['input_ids'] = example['input_ids'][:,:2000]
            example['attention_mask'] = example['attention_mask'][:,:2000]
            example['labels'] = example['labels'][:,:2000]
        example['true_false_labels'] = torch.tensor(batch_con_labels)
        return example
    if args.dataset_name == 'CNNDM':
        return
    if args.dataset_name == 'TWEET':
        return
    if args.dataset_name == 'MEDIA':
        return

class MyTrainer(Trainer):
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        global modules_list
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None

        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    if eval_dataset_name == "test":
                        dataset_metrics = self.evaluate(
                            eval_dataset=eval_dataset
                        )
                        metrics.update(dataset_metrics)
                    else:
                        eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=default_data_collator)
                        all_hidden_states = []
                        all_labels = []
                        model.eval()
                        step = 1
                        i = 1
                        for batch in tqdm(eval_dataloader):
                            with torch.no_grad():
                                input_ids = batch['input_ids']
                                att = batch['attention_mask']
                                input_ids = input_ids.to(model.device)
                                att = att.to(model.device)
                                output = model(input_ids = input_ids, attention_mask = att, output_hidden_states = True)
                                hidden_states = output.hidden_states
                                hidden_states = torch.stack(hidden_states, dim = 0)
                                hidden_states = hidden_states.detach().cpu()
                                #layer batch seq dim
                                all_hidden_states.append(hidden_states[:,:,-1,:].squeeze(2).transpose(0,1))
                                all_labels.append(batch['true_false_labels'])
                                step = step + 1
                                if step == 200:
                                    np.save('probing/opt-'+args.mode+'/all_hidden_states' + str(i) + '.npy', torch.cat(all_hidden_states, dim = 0))
                                    np.save('probing/opt-'+args.mode+'/all_labels' + str(i) + '.npy', torch.cat(all_labels, dim = 0))
                                    all_hidden_states = []
                                    all_labels = []
                                    i = i + 1
                                    step = 1
                        np.save('probing/opt-'+args.mode+'/all_hidden_states' + str(i) + '.npy', torch.cat(all_hidden_states, dim = 0))
                        np.save('probing/opt-'+args.mode+'/all_labels' + str(i) + '.npy', torch.cat(all_labels, dim = 0))
                        all_hidden_states = []
                        all_labels = []
                        acc = np.zeros((LAYER_NUM,1))
                        data = np.concatenate([np.load('probing/opt-'+args.mode+'/all_hidden_states' + str(i+1) + '.npy') for i in range(19)], axis = 0)
                        label = np.concatenate([np.load('probing/opt-'+args.mode+'/all_labels' + str(i+1) + '.npy') for i in range(19)], axis = 0)
                        all_X_train, all_X_val, y_train, y_val = train_test_split(data,label,test_size=0.2,random_state=42)
                        for layer_id in range(LAYER_NUM):
                            X_train = all_X_train[:,layer_id+1,:]
                            X_val = all_X_val[:,layer_id+1,:]
                            clf = LogisticRegression(random_state=42, max_iter=10000).fit(X_train, y_train)
                            y_val_pred = clf.predict(X_val)
                            acc[layer_id][0] = accuracy_score(y_val, y_val_pred)
                        index_list = []
                        for index in range(args.worst_k):
                            index_list.append(np.argmin(acc))
                            acc[np.argmin(acc)][0] = 1.
                        modules_list = []
                        k_proj = []
                        v_proj = []
                        q_proj = []
                        o_proj = []
                        fc1 = []
                        fc2 = []
                        for i in index_list:
                            k_proj.append(f"model.decoder.layers.{i}.self_attn.k_proj.weight")
                            v_proj.append(f"model.decoder.layers.{i}.self_attn.v_proj.weight")
                            q_proj.append(f"model.decoder.layers.{i}.self_attn.q_proj.weight")
                            o_proj.append(f"model.decoder.layers.{i}.self_attn.out_proj.weight")
                            fc1.append(f"model.decoder.layers.{i}.fc1")
                            fc2.append(f"model.decoder.layers.{i}.fc2")
                        modules_list = k_proj + v_proj + q_proj + o_proj + fc1 + fc2
                        with open('/mnt/nas-coai-wlcb/fhwexp/prefix-tuning/opt-modules.json', 'w+') as fs:
                            fs.write(json.dumps({"modules": modules_list})+'\n')
                            fs.close()
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        true_false_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = torch.tensor(0.).to(logits.device)

        if labels is not None and true_false_labels is not None:
            loss_fct_1 = CrossEntropyLoss()
            loss_fct_2 = NLLLoss()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Enable model parallelism
            true_indices = torch.where(true_false_labels == 1)[0]
            false_indices = torch.where(true_false_labels == 0)[0]
            if true_indices.shape[0] != 0:
                true_logits = torch.index_select(shift_logits, dim=0, index=true_indices)
                true_labels = torch.index_select(shift_labels, dim=0, index=true_indices)
                true_logits = true_logits.view(-1, self.config.vocab_size)
                true_labels = true_labels.view(-1)
                true_labels = true_labels.to(true_logits.device)
                loss = loss + loss_fct_1(true_logits, true_labels)
            if false_indices.shape[0] != 0:
                false_logits = torch.index_select(shift_logits, dim=0, index=false_indices)
                false_labels = torch.index_select(shift_labels, dim=0, index=false_indices)
                false_logits = false_logits.view(-1, self.config.vocab_size)
                false_labels = false_labels.view(-1)
                false_labels = false_labels.to(false_logits.device)
                loss = loss - 0.05 * loss_fct_2(torch.softmax(false_logits,dim=-1), false_labels)
            # Flatten the tokens
        # t_y_f_y
        if labels is not None and true_false_labels is None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class OPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)
        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        true_false_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        logits = self.lm_head(outputs[0]).contiguous()

        loss = torch.tensor(0.).to(logits.device)

        #t_y_f_n or con
        if labels is not None and true_false_labels is not None:
            loss_fct_1 = CrossEntropyLoss()
            loss_fct_2 = NLLLoss()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Enable model parallelism
            true_indices = torch.where(true_false_labels == 1)[0]
            false_indices = torch.where(true_false_labels == 0)[0]
            if true_indices.shape[0] != 0:
                true_logits = torch.index_select(shift_logits, dim=0, index=true_indices)
                true_labels = torch.index_select(shift_labels, dim=0, index=true_indices)
                true_logits = true_logits.view(-1, self.config.vocab_size)
                true_labels = true_labels.view(-1)
                true_labels = true_labels.to(true_logits.device)
                loss = loss + loss_fct_1(true_logits, true_labels)
            if false_indices.shape[0] != 0:
                false_logits = torch.index_select(shift_logits, dim=0, index=false_indices)
                false_labels = torch.index_select(shift_labels, dim=0, index=false_indices)
                false_logits = false_logits.view(-1, self.config.vocab_size)
                false_labels = false_labels.view(-1)
                false_labels = false_labels.to(false_logits.device)
                loss = loss - 0.05 * loss_fct_2(torch.softmax(false_logits,dim=-1), false_labels)
            # Flatten the tokens
        # t_y_f_y
        if labels is not None and true_false_labels is None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

class BloomForCausalLM(BloomPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        true_false_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = torch.tensor(0.).to(lm_logits.device)
        if labels is not None and true_false_labels is not None:
            loss_fct_1 = CrossEntropyLoss()
            loss_fct_2 = NLLLoss()
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Enable model parallelism
            true_indices = torch.where(true_false_labels == 1)[0]
            false_indices = torch.where(true_false_labels == 0)[0]
            if true_indices.shape[0] != 0:
                true_logits = torch.index_select(shift_logits, dim=0, index=true_indices)
                true_labels = torch.index_select(shift_labels, dim=0, index=true_indices)
                true_logits = true_logits.view(-1, self.config.vocab_size)
                true_labels = true_labels.view(-1)
                loss = loss + loss_fct_1(true_logits, true_labels)
            if false_indices.shape[0] != 0:
                false_logits = torch.index_select(shift_logits, dim=0, index=false_indices)
                false_labels = torch.index_select(shift_labels, dim=0, index=false_indices)
                false_logits = false_logits.view(-1, self.config.vocab_size)
                false_labels = false_labels.view(-1)
                loss = loss - 0.05 * loss_fct_2(torch.softmax(false_logits,dim=-1), false_labels)
            # Flatten the tokens
        # t_y_f_y
        if labels is not None and true_false_labels is None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)


def main():
    global modules_list
    dataset = load_dataset("json", data_files="/mnt/nas-coai-wlcb/fhwexp/chatgpt-out/XSUM-final.json")
    dataset = dataset["train"].shuffle(seed=42).train_test_split(test_size=0.1)
    # dataset = load_dataset("json", data_files="/mnt/nas-coai-wlcb/fhwexp/chatgpt-out/XSUM-combine-test.json")
    # dataset = dataset["train"].train_test_split(test_size=0.1)
    probing_dataset = load_dataset("json", data_files="/mnt/nas-coai-wlcb/fhwexp/chatgpt-out/defact.json")
    # train_dataset = load_dataset("json", data_files="/mnt/nas-coai-wlcb/fhwexp/chatgpt-out/XSUM-combine-train.json")
    # test_dataset = load_dataset("json", data_files="/mnt/nas-coai-wlcb/fhwexp/chatgpt-out/XSUM-combine-test.json")
    if args.mode == 't_y_f_y':
        train_dataset = dataset["train"].map(batch_preprocessing_t_y_f_y, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
        test_dataset = dataset["test"].map(batch_preprocessing_t_y_f_y, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
        probing_dataset = probing_dataset["train"].map(batch_preprocessing_t_y_f_n, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
    if args.mode == 't_y_f_n':
        train_dataset = dataset["train"].map(batch_preprocessing_t_y_f_n, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
        test_dataset = dataset["test"].map(batch_preprocessing_t_y_f_n, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
    if args.mode == 'con' or args.mode == 'lora':
        train_dataset = dataset["train"].map(batch_preprocessing_con, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
        test_dataset = dataset["test"].map(batch_preprocessing_con, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
        probing_dataset = probing_dataset["train"].map(batch_preprocessing_t_y_f_n, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
    if args.mode == 'procon' or args.mode == 'proun':
        train_dataset = dataset["train"].map(batch_preprocessing_con, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
        test_dataset = dataset["test"].map(batch_preprocessing_con, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
        probing_dataset = probing_dataset["train"].map(batch_preprocessing_t_y_f_n, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=["article","summary","label"], num_proc=1)
    
    training_args = TrainingArguments(
        output_dir='ckpt/opt-'+args.mode+f'/{args.cur_epoch}',
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=True,  # T5 overflows with fp16
        bf16=False,  # Use BF16 if available
        learning_rate=args.lr,
        warmup_ratio=0.2,
        weight_decay=3e-7,
        num_train_epochs=1,
        deepspeed=args.deepspeed,
        # logging & evaluation strategies
        logging_dir="logs/opt-"+args.mode,
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        label_names = label_names,
        load_best_model_at_end=True,
    )
    if args.cur_epoch == 0:
        model = OPTForCausalLM.from_pretrained("../model/"+args.model_name, torch_dtype=torch.float16)
    else:
        fs = open('/mnt/nas-coai-wlcb/fhwexp/prefix-tuning/opt-modules.json', 'r')
        lines = fs.readlines()
        modules_list = json.loads(lines[0])["modules"]
        model = OPTForCausalLM.from_pretrained('/mnt/nas-coai-wlcb/fhwexp/prefix-tuning/ckpt/opt-'+args.mode+f'/{args.cur_epoch-1}/checkpoint-1841', torch_dtype=torch.float16)
    for name, param in model.named_parameters():
        param.requires_grad = False
        if name in modules_list:
            param.requires_grad = True
    data_collator = DefaultDataCollator()
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"probing": probing_dataset, "test": test_dataset},
        data_collator=data_collator
    )
    trainer.train()
    # model = AutoModelForCausalLM.from_pretrained("../model/"+args.model_name, torch_dtype=torch.float16)
    # model = LlamaForCausalLM.from_pretrained("../model/"+args.model_name, torch_dtype=torch.float16)
    
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
if __name__ == "__main__":
    main()
