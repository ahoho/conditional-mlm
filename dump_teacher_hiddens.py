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
from dataclasses import dataclass

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

from data import LineByLinePairedTextDataset, SimpleDataCollator

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
        example["id"] = str(i)
        return example


@dataclass
class DataCollatorForMLMQuery(SimpleDataCollator):
    """
    Sequentially mask each token in a dataset so that we can query the logits

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
        masks = masks.bool()

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


@dataclass
class DataCollatorForSpanMLMQuery(DataCollatorForMLMQuery):
    """
    Sequentially mask each token in a dataset so that we can query the logits

    TODO: Is masking necessary? Can't we just get the outputs of the model on the
    complete, unmasked data?
    """
    num_samples: int = 7 # TODO: change this number?

    def convert_example(self, example):
        """
        Change masking to never split on WordPieces
        """
        # TODO
        # Could improve
        target_mask = example['token_type_ids'].bool()
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

        # Expand masks so that they are always contiguous
        # Slow, but this is only run once
        contig_masks = masks.clone()
        
        # first, look at the 
        tgt_toks = self.tokenizer.convert_ids_to_tokens(tgt)
        word_parts = torch.tensor(
            [
                tgt_toks[i].startswith("##") or tgt_toks[i + 1].startswith("##")
                for i in range(tgt_len - 1)
            ] + [False],
        )
        for mask in contig_masks:
            inside_tokens = torch.where(mask & word_parts)[0]
            for idx in inside_tokens:
                left, right = 1, 1
                while True:
                    if not tgt_toks[idx].startswith("##"):
                        left = 0 # means the masked token is a starting word
                    if left > 0:
                        is_start = not tgt_toks[idx-left].startswith("##")
                        mask[idx-left] = word_parts[idx-left].item() or is_start
                        if is_start:
                            left = 0
                    if right > 0:
                        is_continuing = tgt_toks[idx+right].startswith("##")
                        mask[idx+right] = is_continuing
                        if not is_continuing:
                            right = 0
                    if right == left == 0:
                        break
                    right += 1
                    left += 1

        masks = torch.cat([torch.zeros(self.num_samples, src_len).byte(), masks], dim=1)
        masks = masks.bool()
        contig_masks = torch.cat([torch.zeros(self.num_samples, src_len).byte(), contig_masks], dim=1)
        contig_masks = contig_masks.bool()

        # make BERT inputs
        input_ids = example['input_ids'].repeat((self.num_samples, 1))
        input_ids.masked_fill_(contig_masks, value=self.tokenizer.mask_token_id)
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
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)
    
    all_masks = batch.pop("masks")
    all_masks = [m.to(model.device) for m in all_masks]

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


def build_db_batched(
    source_path,
    target_path,
    out_db,
    model,
    tokenizer,
    truncation,
    batch_size=8,
    output_dtype=np.float32,
    mask_spans=False,
    num_samples=7,
):
    """
    Use the model to give outputs for each target token in the dataset
    """
    dataset = LineByLinePairedTextDatasetWithID(
        tokenizer=tokenizer,
        source_file_path=source_path,
        target_file_path=target_path,
        block_size=tokenizer.max_len,
        truncation=truncation,
    )
    if mask_spans:
        data_collator = DataCollatorForSpanMLMQuery(tokenizer=tokenizer, num_samples=num_samples)
    else:
        data_collator = DataCollatorForMLMQuery(tokenizer=tokenizer, num_samples=num_samples)

    loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=data_collator,
    )
    with tqdm(desc='computing BERT features', total=len(dataset)) as pbar:
        for batch in loader:
            ids = batch.pop("ids")
            outputs = process_batch(batch, model)
            for id, output in zip(ids, outputs):
                out_db[id] = tensor_dumps(output, dtype=output_dtype)
            pbar.update(len(ids))


def main(opts):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # load model & tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path)
    except OSError:
        logger.warning("Tokenizer not in named directory. Checking parent directory")
        tokenizer = AutoTokenizer.from_pretrained(f"{opts.model_name_or_path}/..")

    model = AutoModelForMaskedLM.from_pretrained(opts.model_name_or_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    if opts.fp16:
        if not is_apex_available():
            raise ImportError(
                "Please install apex to use fp16 training."
            )
        model = amp.initialize(model, opt_level="O1") # default level from transformers
    
    # store decoder for quickly producing logits from hidden states
    os.makedirs(opts.output_dir, exist_ok=opts.overwrite)
    torch.save(opts, f"{opts.output_dir}/args.bin")
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
            truncation=opts.truncation_strategy,
            batch_size=opts.batch_size,
            output_dtype=np.float16 if opts.fp16 else np.float32,
            mask_spans=opts.mask_spans,
            num_samples=opts.num_samples,
        )


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
    parser.add_argument('--fp16', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--overwrite', action="store_true", default=False)
    parser.add_argument('--mask_spans', action="store_true", default=False)
    parser.add_argument('--num_samples', type=int, default=7)
    parser.add_argument('--truncation_strategy', default="longest_first")
    args = parser.parse_args()

    main(args)
