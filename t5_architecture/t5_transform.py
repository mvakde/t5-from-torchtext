# /* Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. */
from typing import List, Union

import torch
import torch.nn as nn

# old code
# import torchtext.transforms as T
# from torchtext.data.functional import load_sp_model # redefined using spm
# from torchtext.functional import to_tensor 
# from torchtext.utils import get_asset_local_path # not needed if using spm

# new code -------------------
from typing import Any, Optional
from torch.nn.utils.rnn import pad_sequence

def add_token(input: Any, token_id: Any, begin: bool = True) -> Any:
    """Add token to start or end of sequence

    :param input: Input sequence or batch
    :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    :param token_id: token to be added
    :type token_id: Union[str, int]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    :return: sequence or batch with token_id added to begin or end or input
    :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    """
    if torch.jit.isinstance(input, List[int]) and torch.jit.isinstance(token_id, int):
        if begin:
            return [token_id] + input
        else:
            return input + [token_id]
    elif torch.jit.isinstance(input, List[str]) and torch.jit.isinstance(token_id, str):
        if begin:
            return [token_id] + input
        else:
            return input + [token_id]
    elif torch.jit.isinstance(input, List[List[int]]) and torch.jit.isinstance(token_id, int):
        output: List[List[int]] = []

        if begin:
            for ids in input:
                output.append([token_id] + ids)
        else:
            for ids in input:
                output.append(ids + [token_id])

        return output
    elif torch.jit.isinstance(input, List[List[str]]) and torch.jit.isinstance(token_id, str):
        output: List[List[str]] = []
        if begin:
            for ids in input:
                output.append([token_id] + ids)
        else:
            for ids in input:
                output.append(ids + [token_id])

        return output
    else:
        raise TypeError("Input type not supported")
    
def truncate(input: Any, max_seq_len: int) -> Any:
    """Truncate input sequence or batch

    :param input: Input sequence or batch to be truncated
    :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    :param max_seq_len: Maximum length beyond which input is discarded
    :type max_seq_len: int
    :return: Truncated sequence
    :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    """
    if torch.jit.isinstance(input, List[int]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[str]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[List[int]]):
        output: List[List[int]] = []
        for ids in input:
            output.append(ids[:max_seq_len])
        return output
    elif torch.jit.isinstance(input, List[List[str]]):
        output: List[List[str]] = []
        for ids in input:
            output.append(ids[:max_seq_len])
        return output
    else:
        raise TypeError("Input type not supported")
    
class Sequential(torch.nn.Sequential):
    r"""A container to host a sequence of text transforms."""

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self:
            input = module(input)
        return input

class Truncate(torch.nn.Module):
    r"""Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return truncate(input, self.max_seq_len)
    
class AddToken(torch.nn.Module):
    """Add token to beginning or end of sequence

    :param token: The token to be added
    :type token: Union[int, str]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    """

    def __init__(self, token: Union[int, str], begin: bool = True) -> None:
        super().__init__()
        self.token = token
        self.begin = begin

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """

        return add_token(input, self.token, self.begin)

def to_tensor(input: Any, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> torch.Tensor:
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    :param input: Sequence or batch of token ids
    :type input: Union[List[int], List[List[int]]]
    :rtype: Tensor
    """
    if torch.jit.isinstance(input, List[int]):
        return torch.tensor(input, dtype=torch.long)
    elif torch.jit.isinstance(input, List[List[int]]):
        if padding_value is None:
            output = torch.tensor(input, dtype=dtype)
            return output
        else:
            output = pad_sequence(
                [torch.tensor(ids, dtype=dtype) for ids in input], batch_first=True, padding_value=float(padding_value)
            )
            return output
    else:
        raise TypeError("Input type not supported")

import sentencepiece as spm # new code
def load_sp_model(spm_path): # new code
    r"""Load a  sentencepiece model for file.

    Args:
        spm_path: the file path of the sentencepiece model.

    Outputs:
        output: a SentencePiece model.
    """
    return spm.SentencePieceProcessor(model_file=spm_path)
# ---------------------

class T5Transform(nn.Module):
    """
    This transform makes use of a pre-trained sentencepiece model to tokenize text input. The resulting output is fed to the T5 model.

    Additional details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str
    :param max_seq_len: Maximum sequence length accepted for inputs to T5 model
    :type max_seq_len: int
    :param eos_idx: End-of-sequence token id
    :type eos_idx: int
    :param padding_idx: Padding token id
    :type padding_idx: int

    Example
        >>> from torchtext.prototype.models import T5Transform
        >>> transform = T5Transform("spm_model", max_seq_len = 10, eos_idx = 1, padding_idx = 0)
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: str, max_seq_len: int, eos_idx: int, padding_idx: int):
        super().__init__()
        # old code
        # self.sp_model = load_sp_model(get_asset_local_path(sp_model_path))
        self.sp_model = load_sp_model(sp_model_path) # new code
        self.max_seq_len = max_seq_len
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx
        # old code
        # self.pipeline = T.Sequential(T.Truncate(self.max_seq_len - 1), T.AddToken(token=self.eos_idx, begin=False))
        self.pipeline = Sequential(Truncate(self.max_seq_len - 1), AddToken(token=self.eos_idx, begin=False)) # new code

    def forward(self, input: Union[str, List[str]]) -> torch.Tensor:
        """
        :param input: Input sentence or list of sentences to tokenize.
        :type input: Union[str, List[str]]
        :return: Tokenized text that has been truncated, appended with end-of-sequence token, and padded
        :rtype: torch.Tensor
        """
        tokens = self.encode(input)
        out = to_tensor(self.pipeline(tokens), padding_value=self.padding_idx)
        return out

    @torch.jit.export
    def encode(self, input: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        :param input: Input sentence or list of sentences to tokenize.
        :type input: Union[str, List[str]]
        :return: Tokenized text that has been translated to token ids
        :rtype: Union[List[int], List[List[int]]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[int]] = []
            for text in input:
                tokens.append(self.sp_model.EncodeAsIds(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self.sp_model.EncodeAsIds(input)
        else:
            raise TypeError("Input type not supported")

    @torch.jit.export
    def decode(self, input: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        :param input: List of token ids or list of lists of token ids (i.e. batched).
        :type input: Union[List[int], List[List[int]]]
        :return: Sentence or list of sentencess that were translated from the input token ids
        :rtype: Union[str, List[str]]
        """
        if torch.jit.isinstance(input, List[List[int]]):
            tokens: List[str] = []
            for ids in input:
                tokens.append(self.sp_model.DecodeIds(ids))
            return tokens
        elif torch.jit.isinstance(input, List[int]):
            return self.sp_model.DecodeIds(input)
        else:
            raise TypeError("Input type not supported")
