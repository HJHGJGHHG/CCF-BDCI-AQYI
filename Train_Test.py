import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup


def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(24, -1, -1):
        layer_params = {
            'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        'params': [p for n, p in model.named_parameters() if 'layer_norm' in n or 'linear' in n
                   or 'pooling' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return parameters


def train(args, model, train_iter, val_iter):
    # optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    parameters = get_parameters(model, args.lr, 0.95, 1e-4)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = AdamW(parameters)
    
    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_iter) * args.epochs
    )
    
    for epoch in range(args.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        model = model.train()
        losses = []
        for sample in tqdm(train_iter):
            model.train()
            optimizer.zero_grad()
            input_ids = sample['input_ids'].to(args.device)
            attention_mask = sample['attention_mask'].to(args.device)
            labels = sample['emotions']
            outputs = model(input_ids, attention_mask, labels=labels)
            
            loss = outputs[0]
            losses.append(loss.item())
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        # 一个Epoch训练完毕，输出train_loss
        print('Epoch: {0}   Average Train Loss: {1:>5.6}'.format(epoch + 1, np.mean(losses)))
        eval_model(args, model, val_iter)
    # 训练结束


def eval_model(args, model, val_iter):
    with torch.no_grad():
        model.eval()
        test_loss = []
        for sample in val_iter:
            input_ids = sample['input_ids'].to(args.device)
            attention_mask = sample['attention_mask'].to(args.device)
            labels = sample['emotions']
            outputs = model(input_ids, attention_mask, labels=labels)
            
            test_loss.append(outputs[0].item())
        print('Average Val Loss: {0:>5.6}'.format(np.mean(test_loss)))


def predict(args, model, test_iter):
    def f(a):
        if a > 0.1:
            return a
        return 0
    
    model.eval()
    test_pred = []
    for sample in tqdm(test_iter):
        input_ids = sample['input_ids'].to(args.device)
        attention_mask = sample['attention_mask'].to(args.device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            test_pred.extend(outputs[1].sigmoid().cpu().numpy())
    
    submission = pd.read_csv(args.path + "submit_example.tsv", sep='\t')
    sub = submission.copy()
    sub["emotion"] = test_pred
    sub["emotion"] = sub["emotion"].apply(lambda x: ','.join([str(i) for i in x]))
    
    sub.to_csv(args.path + "submission.tsv", sep='\t', index=False)
