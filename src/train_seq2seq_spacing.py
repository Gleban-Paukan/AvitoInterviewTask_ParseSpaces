#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input  : text_no_spaces
Target : spaced text (ground truth)

Example usage:
python train_seq2seq_spacing.py \
  --train_json data/dataset_1937770_3.csv \
  --model_name ai-forever/ruT5-large \
  --output_dir models/ruT5-large \
  --epochs 5 --batch_size 16 --lr 5e-4 --val_frac 0.1 --seed 42
"""
import argparse, json, math
import torch
from torch.utils.data import Dataset, random_split
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, set_seed)

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = []
    for r in data:
        if not r: 
            continue
        src = r.get('text_no_spaces') or r.get('no_spaces') or r.get('input') or r.get('text')
        tgt = r.get('spaced') or r.get('text_with_spaces') or r.get('target')
        if not (src and tgt):
            splits = r.get('splits')
            if src and splits is not None:
                s = list(src)
                out_str = []
                for i,ch in enumerate(s):
                    out_str.append(ch)
                    if i in set(splits):
                        out_str.append(' ')
                tgt = ''.join(out_str).strip()
        if src and tgt:
            out.append({'src': src, 'tgt': tgt})
    return out

class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_src_len=128, max_tgt_len=128):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        s = self.pairs[idx]['src']
        t = self.pairs[idx]['tgt']
        model_inputs = self.tok(
            s, max_length=self.max_src_len, truncation=True, padding=False, return_tensors="pt"
        )
        with self.tok.as_target_tokenizer():
            labels = self.tok(
                t, max_length=self.max_tgt_len, truncation=True, padding=False, return_tensors="pt"
            )
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        model_inputs['labels'] = labels['input_ids'].squeeze(0)
        return model_inputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json', required=True,)
    ap.add_argument('--model_name', default='ai-forever/ruT5-large')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=20)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_src_len', type=int, default=128)
    ap.add_argument('--max_tgt_len', type=int, default=128)
    ap.add_argument('--freeze_encoder', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)

    data = load_json(args.train_json)
    if len(data) < 50:
        raise RuntimeError(f'Not enough pairs after parsing: {len(data)}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.freeze_encoder:
        if hasattr(model, 'get_encoder'):
            for p in model.get_encoder().parameters():
                p.requires_grad = False

    ds = PairDataset(data, tokenizer, args.max_src_len, args.max_tgt_len)

    val_size = max(1, int(len(ds) * args.val_frac))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest')

    steps_per_epoch = math.ceil(train_size / args.batch_size)
    warmup_steps = max(10, int(0.06 * steps_per_epoch * args.epochs))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        logging_steps=25,
        save_total_limit=2,
        seed=args.seed
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
