import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizer

from span_masking import ParagraphInfo, PairWithSpanMaskingScheme

logger = logging.getLogger(__name__)

class LineByLinePairedTextDataset(Dataset):
    """
    Read in lines from two paired files
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        source_file_path: str,
        target_file_path: str,
        block_size: int,
        truncation: bool = True,
        **kwargs,
    ):
        """
        Read source and target lines from 
        """
        self.tokenizer = tokenizer
        source_lines = self.read_lines_from_file(source_file_path)
        target_lines = self.read_lines_from_file(target_file_path)
        assert(len(source_lines) == len(target_lines))
        
        self.examples = tokenizer(
            source_lines,
            target_lines,
            add_special_tokens=True,
            max_length=block_size,
            truncation=truncation,
        )

    @staticmethod
    def read_lines_from_file(file_path: str) -> List[str]:
        with open(file_path, encoding="utf-8") as f:
            lines = [
                line for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, i) -> Dict[str, List[int]]:
        return {k: torch.tensor(v[i], dtype=torch.long) for k, v in self.examples.items()}


@dataclass
class SimpleDataCollator:
    """
    Modified DataCollatorForLanguageModeling from transformers.data.data_collator
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        return self._tensorize_batch(examples)

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(examples) == 1:
            return examples[0]
        
        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have one."
            )
        batch = {
            k: self._pad_sequence([e[k] for e in examples]) for k in examples[0].keys()
        }
        return batch
    
    def _pad_sequence(self, seq):
        return pad_sequence(
            seq, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )


@dataclass
class MaskingDataCollator(SimpleDataCollator):

    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # NB: this is always a MLM
        batch = self._tensorize_batch(examples)
        return self.mask_tokens(batch)

    def mask_tokens(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 
        80% MASK, 10% random, 10% original.

        Same as DataCollatorForLanguageModeling, but only applies to the `target`
        sequence, as identified by examples['token_type_ids']
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling."
            )
            
        labels = batch['input_ids'].clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Mask the target sentence, not the source
        # (this is the only modification to the original function)
        target_mask = ~batch['token_type_ids'].bool()
        probability_matrix.masked_fill_(target_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        batch['input_ids'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # TODO: Do we still want to maintain this objective?
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        batch['input_ids'][indices_random] = random_words[indices_random]

        batch['labels'] = labels

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return batch


class DummyArgs:
    pass


class SpanBertDataset(LineByLinePairedTextDataset):
    """
    Args:
        dataset (BlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """
    def __init__(
        self,
        tokenizer,
        source_file_path,
        target_file_path,
        block_size,
        truncation=True,
        mlm_probability=0.15,
    ):
        super().__init__(
            tokenizer,
            source_file_path,
            target_file_path,
            block_size,
            truncation=truncation
        )
        self.tokenizer = tokenizer
        self.paragraph_info = ParagraphInfo(tokenizer)

        # TODO: Support these args in training, rather than hard-coding
        args = DummyArgs()
        args.max_pair_targets = 15 # unused
        args.span_lower = 1
        args.span_upper = 10
        args.geometric_p = 0.2
        args.tagged_anchor_prob = 0.
        args.return_only_spans = False
        args.replacement_method = "span"
        args.endpoints = "external"
        args.mask_ratio = mlm_probability

        self.masking_scheme = PairWithSpanMaskingScheme(
            args,
            tokens=list(tokenizer.get_vocab().values()),
            pad=-100, # fills out the rest of the targets/labels
            mask_id=tokenizer.mask_token_id,
            paragraph_info=self.paragraph_info,
        )
        

    def __getitem__(self, index):
        example = {k: np.array(v[index]) for k, v in self.examples.items()}
        inputs = example['input_ids']
        token_type_ids = example['token_type_ids']

        source = inputs[token_type_ids == 0]
        target = inputs[token_type_ids == 1][:-1] # no trailing [SEP]

        masked_block, masked_tgt, _ = self._mask_block(target, tagmap=None)

        masked_inputs = np.concatenate([source, masked_block, [self.tokenizer.sep_token_id]])
        target = np.concatenate([np.full(source.shape[0], -100), masked_tgt, [-100]])
        return {
            "input_ids": torch.tensor(masked_inputs, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask":  torch.tensor(example["attention_mask"], dtype=torch.long),
        }


    def _mask_block(self, sentence, tagmap):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        block = self.masking_scheme.mask(sentence, tagmap)
        return block