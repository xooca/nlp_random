# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
import warnings
from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
)

class QADataModule(pl.LightningDataModule):
    def __init__(self, 
                 model_name_or_path, 
                 train_file, 
                 text_col1,
                 text_col2,
                 label_cols,
                 validation_file, 
                 pad_to_max_length,
                 max_length,
                 doc_stride,
                 preprocessing_num_workers=4, 
                 overwrite_cache=False, 
                 train_batch_size=20, 
                 val_batch_size=20, 
                 dataloader_num_workers=4):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.model_name_or_path = model_name_or_path
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.text_col1 = text_col1
        self.text_col2 = text_col2
        self.label_cols = label_cols
        self.max_length = max_length
        self.doc_stride = doc_stride

    def setup(self, stage):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        datasets = load_dataset("csv", data_files=data_files)

        column_names = datasets["train"].column_names

        # When using line_by_line, we just tokenize each nonempty line.
        self.padding = "max_length" if self.pad_to_max_length else False

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=datasets["train"].column_names,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=not self.overwrite_cache,            
        )
        
        data_collator = DefaultDataCollator()

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )
    
    @staticmethod
    def tokenize_function(examples):
        # Tokenize our examples with truncation and padding, but keep the overflows using a
        # stride. This results in one example possible giving several features when a context is long,
        # each of those features having a context that overlaps a bit the context of the previous
        # feature.

        examples[self.text_col1] = [q.lstrip() for q in examples[self.text_col1]]
        examples[self.text_col2] = [c.lstrip() for c in examples[self.text_col2]]
        ans = []
        for val in examples[self.label_col]:
            ans.append(ast.literal_eval(val))
        examples[self.label_col] = ans

        tokenized_examples = tokenizer(
            examples[self.text_col1],
            examples[self.text_col2],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )
        overflow_to_sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
        offset_mapping = tokenized_examples["offset_mapping"]
        ans=[]
        for i, offsets in enumerate(overflow_to_sample_mapping):
            ans.append(examples[self.label_col][0])
        tokenized_examples[self.label_col] = ans
        return tokenized_examples
    
    @staticmethod
    def prepare_train_features(examples):
        examples[self.text_col1] = [q.lstrip() for q in examples[self.text_col1]]
        examples[self.text_col2] = [c.lstrip() for c in examples[self.text_col2]]

        tokenized_examples = tokenizer(
            examples[self.text_col1],
            examples[self.text_col2],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )
        overflow_to_sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
        offset_mapping = tokenized_examples["offset_mapping"]
        for label_col in label_cols:
            ans={}
            for i, offsets in enumerate(overflow_to_sample_mapping):
                ans.update({label_col: examples[label_col][0]})
        tokenized_examples['labels'] = ans
        return tokenized_examples