#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для инференса модели восстановления пробелов.

Выводит два файла:
- JSONL: {"id", "spaced", "splits"} для каждой строки
- TXT: строки с восстановленными пробелами

Пример запуска:
python infer_seq2seq_spacing.py --model_dir models/byt5-spacer \
  --in_file data/test.csv --in_format csv \
  --out_json data/pred.jsonl --out_txt data/pred.txt
"""
import argparse, os, json, re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def read_inputs(path: str, in_format: str):
    """Чтение входных данных (поддерживается CSV).
    Каждая строка: id,text_no_spaces
    Возвращает список (id, text).
    """
    items = []
    if in_format == 'csv':
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            first = True
            for i, line in enumerate(f):
                s = line.rstrip("\r\n")
                if not s.strip():
                    continue
                if first:
                    left = s.split(",", 1)[0]
                    if (not left.isdigit()) and ('text' in s.lower()):
                        first = False
                        continue
                    first = False
                if "," not in s:
                    items.append((i, s.strip()))
                    continue
                left, right = s.split(",", 1)
                left = left.strip()
                try:
                    rid = int(left)
                except ValueError:
                    rid = left
                text = right.strip()
                if text:
                    items.append((rid, text))
        return items
    else:
        raise ValueError('Unsupported in_format')

def spaced_to_splits(no_space: str, spaced: str):
    """Вычисляем позиции вставленных пробелов.
    Возвращает список индексов, после каких символов был вставлен пробел.
    """
    idx = 0; splits = []
    for ch in spaced:
        if ch.isspace():
            if idx>0: splits.append(idx-1)
            continue
        if idx >= len(no_space):
            break
        idx += 1
    return splits

def main():
    ap = argparse.ArgumentParser()
    # пути и параметры модели/файлов
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--in_file', required=True)
    ap.add_argument('--in_format', choices=['csv'], default='csv') # поддерживается только csv
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--out_txt', required=True)
    # гиперпараметры генерации
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_beams', type=int, default=4)
    ap.add_argument('--no_repeat_ngram_size', type=int, default=3)
    ap.add_argument('--repetition_penalty', type=float, default=1.2)
    ap.add_argument('--length_penalty', type=float, default=0.8)
    ap.add_argument('--encoder_no_repeat_ngram_size', type=int, default=0)

    args = ap.parse_args()

    # загружаем модель и токенайзер
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    # читаем входные данные
    items = read_inputs(args.in_file, args.in_format)
    outs = []
    texts = [t for _, t in items]
    ids = [i for i, _ in items]

    # динамический лимит на число новых токенов (~25% от длины входа)
    def dyn_max_new(src: str, cap: int) -> int:
        return min(cap, max(8, int(len(src) * 0.25)))

    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True).to(device)

        max_new = [dyn_max_new(s, args.max_new_tokens) for s in batch]
        max_new_tokens = int(max(max_new) if max_new else args.max_new_tokens)

        # генерация с beam search
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=args.encoder_no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                early_stopping=True
            )
        # раскодируем токены в строки
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for rid, src, pred in zip(ids[start:start+args.batch_size], batch, dec):
            spaced = re.sub(r'\s+', ' ', pred).strip()  # чистим лишние пробелы
            splits = spaced_to_splits(src, spaced)      # восстанавливаем индексы вставок
            outs.append({'id': int(rid), 'spaced': spaced, 'splits': splits})

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        for r in outs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(args.out_txt, 'w', encoding='utf-8') as f:
        for r in outs:
            f.write(r['spaced'] + '\n')

if __name__ == '__main__':
    main()
