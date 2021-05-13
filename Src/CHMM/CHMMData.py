import torch
import numpy as np

from typing import List, Optional
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: List[str],
                 embs: List[torch.Tensor],
                 obs: List[List[int]],
                 lbs: Optional[List[List[str]]] = None
                 ):
        """
        A wrapper class to create syntax dataset for syntax expansion training.
        """
        super().__init__()
        self._embs = embs
        self._obs = obs
        self._text = text
        self._lbs = lbs

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._obs)

    @property
    def embs(self):
        return self._embs

    @property
    def text(self):
        return  self._text

    @property
    def lbs(self):
        return self._lbs

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, value):
        print('[Warning] the value of observations has been changed')
        self._obs = value

    @embs.setter
    def embs(self, value):
        print('[Warning] the value of embeddings has been changed')
        self._embs = value

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._lbs is not None:
            return self._text[idx], self._embs[idx], self._obs[idx], self._lbs[idx]
        else:
            return self._text[idx], self._embs[idx], self._obs[idx]


def batch_prep(emb_list: List[torch.Tensor],
               obs_list: List[torch.Tensor],
               txt_list: Optional[List[List[str]]] = None,
               lbs_list: Optional[List[dict]] = None):
    """
    Pad the instance to the max seq max_seq_length in batch
    """
    for emb, obs, txt in zip(emb_list, obs_list, txt_list):
        assert len(obs) + 1 == len(emb) == len(txt)
    d_emb = emb_list[0].size(-1)
    _, n_src, n_obs = obs_list[0].size()
    seq_lens = [len(obs)+1 for obs in obs_list]
    max_seq_len = np.max(seq_lens)

    emb_batch = torch.stack([
        torch.cat([inst, torch.zeros([max_seq_len-len(inst), d_emb])], dim=-2) for inst in emb_list
    ])

    prefix = torch.zeros([1, n_src, n_obs])
    prefix[:, :, 0] = 1
    obs_batch = torch.stack([
        torch.cat([prefix.clone(), inst, prefix.repeat([max_seq_len-len(inst)-1, 1, 1])])
        for inst in obs_list
    ])
    obs_batch /= obs_batch.sum(dim=-1, keepdim=True)

    # increment the indices of the true spans
    lbs_batch = [{(i+1, j+1): v for (i, j), v in lbs.items()} for lbs in lbs_list] \
        if lbs_list is not None else None

    # obs_batch = torch.tensor(obs_batch, dtype=torch.float)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    return emb_batch, obs_batch, seq_lens, txt_list, lbs_batch


def collate_fn(insts):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    all_insts = list(zip(*insts))
    if len(all_insts) == 4:
        txt, embs, obs, lbs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt, lbs_list=lbs)
    elif len(all_insts) == 3:
        txt, embs, obs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt)
    else:
        raise ValueError
    return batch
