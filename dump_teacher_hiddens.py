"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

precompute hidden states of CMLM teacher to speedup KD training
"""
import argparse
import io
import logging
import os
import shelve

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from transformers.file_utils import is_apex_available

from data import LineByLinePairedTextDataset, DataCollatorForConditionalMLM

if is_apex_available():
    from apex import amp

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def tensor_dumps(tensor, dtype=np.float16):
    with io.BytesIO() as writer:
        np.save(writer, tensor.cpu().numpy().astype(dtype),
                allow_pickle=False)
        dump = writer.getvalue()
    return dump


def gather_hiddens(hiddens, masks):
    outputs = []
    for hid, mask in zip(hiddens.split(1, dim=1), masks.split(1, dim=1)):
        if mask.sum().item() == 0:
            continue
        mask = mask.unsqueeze(-1).expand_as(hid)
        outputs.append(hid.masked_select(mask))
    output = torch.stack(outputs, dim=0)
    return output


class LineByLinePairedTextDatasetWithID(LineByLinePairedTextDataset):
    def __getitem__(self, i):
        example = super().__getitem__(i)
        example["id"] = i
        return example


class DataCollatorForMLMQuery(DataCollatorForConditionalMLM):
    """
    Sequentially mask each word in a dataset so that we can query the 

    TODO: Is masking necessary? Can't we just get the outputs of the model on the
    complete, unmasked data?
    """
    num_samples: int = 7

    def __call__(self, examples):
        batch_data = [self.convert_example(e) for e in examples]
        return self._tensorize_batch(batch_data)

    def _tensorize_batch(self, examples):
        if len(examples) == 1:
            return examples[0]

        # unpack the list of dictionary into a dictionary of tensors
        batch = {
            "ids": [e["id"] for e in examples],
            "input_ids": self._pad_sequence([e["input_ids"] for e in examples]),
            "token_type_ids": self._pad_sequence([e["token_type_ids"] for e in examples]),
            "attention_mask":  self._pad_sequence([e["attention_mask"] for e in examples]),
            "masks": [e["masks"] for e in examples],
        }
        return batch

    def _pad_sequence(self, seq):
        """
        Sequences are now matrices, and we can't use the default pad_sequence
        """
        batch_size = sum(s.size(0) for s in seq)
        max_len = max(s.size(1) for s in seq)
        padded_batch = torch.full(
            (batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        i = 0
        for data in seq:
            block, length = data.size()
            padded_batch[i:i+block, :length] = data
            i+= block
        return padded_batch
    
    def convert_example(self, example):
        """
        Modified version of original `convert_example` to accomodate tensors & 
        new transformers library. Masks all words in an example
        """
        target_mask = example['token_type_ids'].bool()
        src = example['input_ids'][~target_mask]
        tgt = example['input_ids'][target_mask]

        # build the random masks
        tgt_len = example['token_type_ids'].sum().item()
        src_len = len(example['token_type_ids']) - tgt_len
        if tgt_len <= self.num_samples:
            masks = torch.eye(tgt_len).byte()
            self.num_samples = tgt_len
        else:
            mask_inds = [list(range(i, tgt_len, self.num_samples))
                        for i in range(self.num_samples)]
            masks = torch.zeros(self.num_samples, tgt_len).byte()
            for i, indices in enumerate(mask_inds):
                for j in indices:
                    masks.data[i, j] = 1
        assert (masks.sum(dim=0) != torch.ones(tgt_len).long()).sum().item() == 0
        assert masks.sum().item() == tgt_len
        masks = torch.cat([torch.zeros(self.num_samples, src_len).byte(), masks], dim=1)

        # make BERT inputs
        input_ids = example['input_ids'].repeat((self.num_samples, 1))
        input_ids.masked_fill_(masks, value=self.tokenizer.mask_token_id)
        attention_mask = example['attention_mask'].repeat((self.num_samples, 1))
        token_type_ids = example['token_type_ids'].repeat((self.num_samples, 1))

        return {
            "id": example["id"],
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "masks": masks,
        }


def process_batch(batch, model):
    """
    Pass the batch through the model
    """
    all_masks = batch.pop("masks")

    if model.config.model_type == "bert":
        outputs = model.bert(**batch)
        sequence_output = outputs[0]
        hiddens = model.cls.predictions.transform(sequence_output) # first layer of LM head
    if model.config.model_type == "roberta":
        hiddens = model.roberta(**batch)
        sequence_output = outputs[0]
        hiddens = model.lm_head.dense(sequence_output)
        hiddens = model.lm_head.layer_norm(sequence_output)
        
    i = 0
    all_hiddens = []
    for masks in all_masks:
        block, len_ = masks.size()
        hids = hiddens[i:i+block, :len_, :]
        all_hiddens.append(gather_hiddens(hids, masks))
        i += block
    return all_hiddens


def build_db_batched(source_path, target_path, out_db, model, tokenizer, batch_size=8):
    """
    Use the model to give outputs for each target token in the dataset
    """
    dataset = LineByLinePairedTextDatasetWithID(
        tokenizer, source_path, target_path, tokenizer.max_len
    )
    data_collator = DataCollatorForMLMQuery(tokenizer=tokenizer, num_samples=7)
    loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=data_collator
    )
    with tqdm(desc='computing BERT features', total=len(dataset)) as pbar:
        for batch in loader:
            ids = batch.pop("ids")
            outputs = process_batch(batch, model)
            for id, output in zip(ids, outputs):
                out_db[id] = tensor_dumps(output)
            pbar.update(len(ids))


def main(opts):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path)

    model = AutoModelWithLMHead.from_pretrained(opts.model_name_or_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    if self.args.fp16:
        if not is_apex_available():
            raise ImportError(
                "Please install apex to use fp16 training."
            )
        model = amp.initialize(model, opt_level="O1") # default level from transformers
    
    # store decoder for quickly producing logits from hidden states
    if model.config.model_type == "bert":
        torch.save(model.cls.predictions.decoder, f"{opts.output_dir}/linear.pt")
    if model.config.model_type == "roberta":
        torch.save(model.lm_head.decoder, f"{opts.output_dir}/linear.pt")

    with shelve.open(f"{opts.output_dir}/db") as out_db, torch.no_grad():
        build_db_batched(
            source_path=opts.source_path,
            target_path=opts.target_path,
            out_db=out_db,
            model=model,
            tokenizer=tokenizer,
            batch_size=opts.batch_size,
        )

    # create DB
    with shelve.open(f'{opts.output}/db') as out_db, \
            torch.no_grad():
        build_db_batched(opts.db, out_db, bert, toker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        required=True,
        help='Model checkpoint for weights initialization'
    )
    parser.add_argument('--source_path', required=True, help="Location of source docs")
    parser.add_argument('--target_path', required=True, help="Location of target docs")
    parser.add_argument('--output_dir', required=True, help='Dump output to this dir')
    parser.add_argument('--fp16', actions="store_true", default=False)
    parser.add_argument('--batch_size', )
    args = parser.parse_args()

    main(args)
