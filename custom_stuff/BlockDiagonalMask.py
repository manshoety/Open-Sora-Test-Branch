import torch
import torch.nn as nn
import math
from typing import Optional, Sequence, Tuple, Union

class BlockDiagonalMask:
    def __init__(self, q_seqinfo, k_seqinfo, batch_sizes: Optional[Sequence[int]] = None):
        self.q_seqinfo = q_seqinfo
        self.k_seqinfo = k_seqinfo
        self._batch_sizes = batch_sizes

    def _create_block_mask(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        assert shape[-1] == self.k_seqinfo.seqstart_py[-1], (shape[-1], self.k_seqinfo.seqstart_py[-1])
        assert shape[-2] == self.q_seqinfo.seqstart_py[-1], (shape[-2], self.q_seqinfo.seqstart_py[-1])
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(self.q_seqinfo.intervals(), self.k_seqinfo.intervals())):
            mask[q_start:q_end, k_start:k_end] = self._create_block_mask((q_end - q_start, k_end - k_start), dtype=dtype, device=device)
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)

    @classmethod
    def from_seqlens(cls, q_seqlen: Sequence[int], kv_seqlen: Optional[Sequence[int]] = None) -> "BlockDiagonalMask":
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        if kv_seqlen is None or q_seqlen == kv_seqlen:
            k_seqinfo = q_seqinfo
        else:
            k_seqinfo = _SeqLenInfo.from_seqlens(kv_seqlen)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    @classmethod
    def from_tensor_list(cls, tensors: Sequence[torch.Tensor]) -> Tuple["BlockDiagonalMask", torch.Tensor]:
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        seqlens = []
        for x in tensors:
            for _ in range(x.shape[0]):
                seqlens.append(x.shape[1])
        block_diag = cls.from_seqlens(seqlens)
        block_diag._batch_sizes = batch_sizes
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in tensors)
        concat_tensors = torch.cat(tensors_bs1, dim=1)
        return block_diag, concat_tensors

    @classmethod
    def from_tensor_lists_qkv(cls, tensors_q: Sequence[torch.Tensor], tensors_k: Sequence[torch.Tensor], tensors_v: Optional[Sequence[torch.Tensor]] = None) -> Tuple["BlockDiagonalMask", torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert len(tensors_q) == len(tensors_k)
        assert tensors_v is None or len(tensors_v) == len(tensors_q)
        batch_sizes = [tensor.shape[0] for tensor in tensors_q]
        q_seqlens, kv_seqlens = [], []
        for i, (q, k) in enumerate(zip(tensors_q, tensors_k)):
            assert q.shape[0] == k.shape[0]
            q_seqlens += [q.shape[1]] * q.shape[0]
            kv_seqlens += [k.shape[1]] * k.shape[0]
            assert tensors_v is None or tensors_v[i].shape[:2] == k.shape[:2]
        block_diag = cls.from_seqlens(q_seqlens, kv_seqlens)
        block_diag._batch_sizes = batch_sizes
        return (
            block_diag,
            torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_q], dim=1),
            torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_k], dim=1),
            torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_v], dim=1) if tensors_v is not None else None,
        )

    def split_queries(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def split_kv(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.k_seqinfo.split(tensor, self._batch_sizes)

    def split(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        assert self.q_seqinfo is self.k_seqinfo
        return self.q_seqinfo.split(tensor, self._batch_sizes)

class _SeqLenInfo:
    @classmethod
    def from_seqlens(cls, seqlens: Sequence[int]) -> "_SeqLenInfo":
        instance = cls()
        instance.seqstart_py = torch.cumsum(torch.tensor([0] + seqlens[:-1]), dim=0).tolist()
        instance.seqlens_py = seqlens
        return instance

    def intervals(self):
        return [(start, start + length) for start, length in zip(self.seqstart_py, self.seqlens_py)]

    def split(self, tensor: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None) -> Sequence[torch.Tensor]:
        if batch_sizes is None:
            split_sizes = self.seqlens_py
        else:
            split_sizes = [self.seqlens_py[i] for i in range(len(self.seqlens_py)) for _ in range(batch_sizes[i])]
        return torch.split(tensor, split_sizes, dim=1)