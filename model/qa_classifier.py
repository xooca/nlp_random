import numpy as np
import pandas as pd
from pathlib import Path

import os
import random

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from collections import Counter
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AlbertModel, AlbertTokenizer,RobertaForQuestionAnswering,BertPreTrainedModel,RobertaTokenizer
import tokenizers
import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaForQuestionAnswering,RobertaConfig,RobertaModel,RobertaForMaskedLM
import torch

from tqdm import tqdm_notebook as tqdm
import itertools
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
##https://towardsdatascience.com/multilabel-classification-with-pytorch-in-5-minutes-a4fa8993cbc7
class MultiLabelBase(pl.LightningModule):
    def __init__(self, model, tokenizer, prediction_save_path,class_num,label_cols):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prediction_save_path = prediction_save_path
        self.drop_out = nn.Dropout(0.1)
        #self.l0 = nn.Linear(1024 * 2, class_num)
        self.class_num = class_num
        self._init_weights(module = self.l0)
        #torch.nn.init.normal_(self.l0.weight, std=0.02)
        self.loss_average_val =0
        self.loss_average_train = 0
        self.label_cols = label_cols
        self.multi_label_output(in_features=1024*2)

    def multi_label_output(self,in_features):
        self.label_output = {}
        for col in self.label_cols:
            self.label_output.update({col:nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=in_features, out_features=1)
            )})
        
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_device(self):
        return self.bertmodel.state_dict()['bert.embeddings.word_embeddings.weight'].device

    def save_predictions(self, idx,start_positions, end_positions,filtered_output):
        d = pd.DataFrame({'text_ID':idx,'start_position':start_positions, 'end_position':end_positions,'selected_text':filtered_output})
        d.to_csv(self.prediction_save_path, index=False)
        
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.model(
            ids,
            mask,
            token_type_ids=token_type_ids
        ) 
        out = torch.cat((out[-1], out[-2]), dim=-1) 
        out = self.drop_out(out) 
        #logits = self.l0(out) 
        fwd_return = {}
        for label_col,output in self.label_output.items():
            fwd_return.update({'label_col':output(out)})
        #start_logits, end_logits = logits.split(1, dim=-1)
        #start_logits = start_logits.squeeze(-1) 
        #nd_logits = end_logits.squeeze(-1) 
        return fwd_return
    
    def criterion(self,loss_func,predictions,actuals):
        losses = 0
        for i, key in enumerate(predictions):
            losses += loss_func(predictions[key], actuals['labels'][key].to(device))
        return losses

    def loss(self,predictions, actuals):
        """
        Return the sum of the cross entropy losses for both the start and end logits
        """
        loss_fct = nn.CrossEntropyLoss()
        return self.criterion(loss_func,predictions,actuals)
    
    def training_step(self, batch, batch_number):
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        idx = batch['idx']
        #ids, mask, token_type_ids
        predictions = self.forward(batch['ids'],batch['mask'],batch['token_type_ids'])
        loss = self.loss(predictions,batch['labels'])
        self.loss_average_train = (self.loss_average_train + loss)/(batch_number+1)
        if batch_number % 200 == 0: 
            print(f"TRAIN : Batch {batch_number} Average Loss {self.loss_average_train} batch loss: {loss}")
        return {'loss':loss, 'idx':idx,'total_average_train_loss':self.loss_average_train}

    def validation_step(self, batch, batch_number):
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        idx = batch['idx']
        predictions = self.forward(batch['ids'],batch['mask'],batch['token_type_ids'])
        loss = self.loss(predictions,batch['label'])
        self.loss_average_val = (self.loss_average_val + loss)/(batch_number+1)
        if batch_number % 100 == 0: 
            print(f"VAL : Batch {batch_number} Average Loss {self.loss_average_val} batch loss: {loss}")
        return {'loss':loss, 'idx':idx,'total_average_val_loss':self.loss_average_val}

    def training_end(self, outputs):
        """
        outputs(dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
        #train_num_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
        #l = outputs['loss']
        #tl = outputs['total_train_loss']/train_num_steps
        #print(f"TRAIN STEP END : Total Return Loss {l}, TOTAL LOSS {tl}, TRAIN STEPS : {train_num_steps}")
        return {'loss':outputs['loss']}

    def validation_end(self, outputs):
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """        
        return {'loss':torch.mean(torch.tensor([output['loss'] for output in outputs])).detach()}

    def test_step(self, batch, batch_number):
        """
        (batch) -> (dict or OrderedDict)
        """
        idx = batch['idx']
        predicted_classes = self.forward(batch['ids'],batch['mask'],batch['token_type_ids'])
        target = batch['target']
        offsets = batch['offsets']
        return {'predicted_classes':predicted_classes, 'target':target, 'idx':idx}
    
    def test_end(self, outputs):
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """
        start_scores = torch.cat([output['start_scores'] for output in outputs]).detach().cpu().numpy()
        start_positions = np.argmax(start_scores, axis=1) - 1

        end_scores = torch.cat([output['end_scores'] for output in outputs]).detach().cpu().numpy()
        end_positions = np.argmax(end_scores, axis=1) - 1
        idx = [output['idx'] for output in outputs]
        idx = list(itertools.chain.from_iterable(idx))
        
        tweet = [output['tweet'] for output in outputs]
        tweet = list(itertools.chain.from_iterable(tweet))

        offsets = [output['offsets'] for output in outputs]
        offsets = list(itertools.chain.from_iterable(offsets))
        
        sentiment = [output['sentiment'] for output in outputs]
        sentiment = list(itertools.chain.from_iterable(sentiment))
        
        filtered_output= self.extract_selected_text_batch(tweet, sentiment,start_positions, end_positions,offsets)
        self.save_predictions(idx,start_positions, end_positions,filtered_output)
        return {}

    #def configure_optimizers(self):
    #    return optim.Adam(self.parameters(), lr=2e-5)
    
    def configure_optimizers(self):
        num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_train_steps)
        return [optimizer], [scheduler]


    @pl.data_loader
    def train_dataloader(self):
        pass

    @pl.data_loader
    def val_dataloader(self):
        pass

    @pl.data_loader
    def test_dataloader(self):
        pass