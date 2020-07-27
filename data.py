import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizer

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
        sep_token: str,
        block_size: int,
    ):
        """
        Read source and target lines from 
        """
        source_lines = self.read_lines_from_file(source_file_path)
        target_lines = self.read_lines_from_file(target_file_path)
        assert(len(source_lines) == len(target_lines))
        
        self.examples = tokenizer(
            source_lines,
            target_lines,
            add_special_tokens=True,
            padding=True, # TODO: should this be done in the collator?
            truncation=True,
            max_length=block_size,
            return_tensors='pt',
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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.examples.items()}

@dataclass
class DataCollatorForConditionalMLM:
    """
    Modified DataCollatorForLanguageModeling from transformers.data.data_collator
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # NB: this is always a MLM
        return self.mask_tokens(examples)

    def mask_tokens(self, examples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

        labels = examples['input_ids'].clone()
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
        target_mask = examples['token_type_ids'].bool()
        probability_matrix.masked_fill_(target_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        examples['input_ids'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # TODO: Do we still want to maintain this objective?
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        examples['input_ids'][indices_random] = random_words[indices_random]

        examples['labels'] = labels

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return examples