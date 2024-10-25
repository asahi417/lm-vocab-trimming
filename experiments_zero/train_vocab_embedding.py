"""Embedding trainer."""
import os
import logging
import random
from math import prod
from typing import Dict, Any, Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Trainer:

    def __init__(self,
                 output_dir: str,
                 dataset_name: str = "sentence-transformers/parallel-sentences-ccmatrix",
                 dataset_config: str = "en-zh",
                 dataset_split: str = "train",
                 dataset_column_target: str = "english",
                 dataset_column_source: str = "non_english",
                 model_target: str = "roberta-base",
                 model_source: str = "roberta-base",
                 attn_implementation: str = "sdpa",
                 parameter_prefix: str = "embeddings",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 random_seed: int = 42,
                 weight_decay: float = 1e-7,
                 lr: float = 0.00002,
                 lr_warmup: int = 100):
        # config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger()
        file_handler = logging.FileHandler(f'{self.output_dir}/training.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        self.logger.addHandler(file_handler)
        # model
        model_config = dict(torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        self.model_target = AutoModelForSequenceClassification.from_pretrained(model_source, **model_config)
        self.model_target.train()
        self.tokenizer_target = AutoTokenizer.from_pretrained(model_target)
        self.model_source = AutoModelForSequenceClassification.from_pretrained(model_source, **model_config)
        self.model_source.eval()
        self.tokenizer_source = AutoTokenizer.from_pretrained(model_source)
        # dataset
        self.dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        self.dataset_column_target = dataset_column_target
        self.dataset_column_source = dataset_column_source
        self.logger.info(f"[dataset size] {len(self.dataset):,}")
        # fix random seed
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        # optimizers
        params = [p for n, p in self.model_source.named_parameters() if n.startswith(parameter_prefix)]
        self.optimizer = torch.optim.AdamW([{"params": params, "weight_decay": weight_decay}], lr=lr)
        # scheduler

        def lr_lambda(current_step: int):
            current_step += 1
            if current_step < lr_warmup:
                return float(current_step) / float(max(1, lr_warmup))
            return 1

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)
        self.logger.info(f"[learnable parameters] {int(sum([prod(i.shape) for i in params])):,}")
        # device
        self.parallel = torch.cuda.device_count() > 1
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        if self.parallel:
            self.model_target = torch.nn.DataParallel(self.model_target)
            self.model_source = torch.nn.DataParallel(self.model_source)
        self.model_target.to(self.device)
        self.model_source.to(self.device)
        self.logger.info(f"[device info] parallel={self.parallel}, device={self.device}")

    def unwrap(self, model):
        if self.parallel:
            return model.module
        return model

    def train(self, batch_size: int, epoch: int, max_length: int = 256, temperature: float = 1, log_interval: int = 10, repo_id: Optional[str]=None) -> None:
        total_step = int(len(self.dataset)/batch_size)
        for e in range(epoch):
            self.dataset = self.dataset.shuffle(self.random_seed)
            data_loader = self.dataset.iter(batch_size=batch_size, drop_last_batch=True)
            loss = 0
            for b, batch in enumerate(data_loader):
                loss += self.single_step(batch, max_length, temperature)
                if b % log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"[epoch={e + 1}/{epoch}, batch={b + 1}/{total_step}] loss={loss/log_interval}, lr={lr}")
                    loss = 0
            self.unwrap(self.model_source).save_pretrained(f"{self.output_dir}/epoch_{e}")
            self.tokenizer_source.save_pretrained(f"{self.output_dir}/epoch_{e}")
        if repo_id:
            self.unwrap(self.model_source).push_to_hub(repo_id)
            self.tokenizer_source.push_to_hub(repo_id)

    def single_step(self, batch: Dict[str, Any], max_length: int = 256, temperature: float = 1) -> float:
        self.optimizer.zero_grad()
        encode_config = dict(return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')
        # get last hidden state of the target model
        encode_target = self.tokenizer_target(batch[self.dataset_column_target], **encode_config)
        encode_target["inputs_embeds"] = self.unwrap(self.model_target).embeddings(encode_target.pop("input_ids"))
        with torch.no_grad():
            output_target = self.model_target(**{k: v.to(self.device) for k, v in encode_target}, output_hidden_states=True)
            embedding_target = output_target.hidden_states[-1].mean(1)  # batch x dim
            # get last hidden state of the source model
            encode_source = self.tokenizer_source(batch[self.dataset_column_source], encode_config)
            output_source = self.model_source(**{k: v.to(self.device) for k, v in encode_source}, output_hidden_states=True)
            embedding_source = output_source.hidden_states[-1].mean(1)  # batch x dim
        # compute NCE loss
        cos_sim = torch.nn.CosineSimilarity(dim=3)
        distance = torch.exp(cos_sim(embedding_target.unsqueeze(1), embedding_source.unsqueeze(0))/temperature)
        logit_p = torch.diagonal(distance, 0)
        denominator = torch.sum(torch.tril(distance, diagonal=-1))
        loss = torch.sum(- torch.log(logit_p / (denominator.unsqueeze(-1) + logit_p + 1e-5)))
        # backprop
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.cpu().item()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Contrastive training of new embedding.')
    parser.add_argument('-t', '--model-target', type=str, default="roberta-base")
    parser.add_argument('-s', '--model-source', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('--repo-id', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, default="sentence-transformers/parallel-sentences-ccmatrix")
    parser.add_argument('-c', '--dataset-config', type=str, default="en-zh")
    parser.add_argument('--dataset-split', type=str, default="train")
    parser.add_argument('--dataset-column-target', type=str, default="english")
    parser.add_argument('--dataset-column-source', type=str, default="non_english")
    parser.add_argument('--attn-implementation', type=str, default="sdpa")
    parser.add_argument('--parameter-prefix', type=str, default="embeddings")
    parser.add_argument('--torch-dtype', type=str, default="torch.bfloat16")
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--lr-warmup', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--log-interval', type=int, default=1)
    opt = parser.parse_args()
    trainer = Trainer(
        output_dir=opt.output_dir,
        dataset_name=opt.dataset_name,
        dataset_config=opt.dataset_config,
        dataset_split=opt.dataset_split,
        dataset_column_target=opt.dataset_column_target,
        dataset_column_source=opt.dataset_column_source,
        model_target=opt.model_target,
        model_source=opt.model_source,
        attn_implementation=opt.attn_implementation,
        parameter_prefix=opt.parameter_prefix,
        torch_dtype=eval(opt.torch_dtype),
        random_seed=opt.random_seed,
        weight_decay=opt.weight_decay,
        lr=opt.lr,
        lr_warmup=opt.lr_warmup
    )
    trainer.train(
        batch_size=opt.batch_size,
        max_length=opt.max_length,
        epoch=opt.epoch,
        log_interval=opt.log_interval,
        repo_id=opt.repo_id
    )