# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
from torch.autograd import Variable
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch
import numpy as np

@dataclass
class NCECriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

import pickle
with open("de_freq_dict.pickle", "rb") as f:
    global de_freq_dict
    de_freq_dict = pickle.load(f) # id to freq
    # import pdb; pdb.set_trace()
    # print('')
    



def compute_mu(logits, neg_targets, pos_logits, neg_i):
    """
    Compute mu for each negative sample in batch (bs x 1 x seq_len).


    Note: the order doesn't matter as the loss will be summed up any way and the neg logits are randomly selected.


    Returns:
        mu and its corresponding negative logits
    """
    neg_logits_i = logits.gather(dim=-1,index=torch.unsqueeze(neg_targets[:,neg_i, :], -1)).squeeze(-1)  # bs x seq_len
    mu_i = torch.exp(neg_logits_i-pos_logits)
    return neg_logits_i, mu_i

def validate_samples(neg_targets, target_batch):
    """
    neg_targets    bs x (k_neg) x seq_len
    target_batch   bs x seq_len
    k_neg_dim: whether k_neg dimension is in neg_targets
    """
    is_k_neg_dim = len(neg_targets.shape) == 3
    if is_k_neg_dim:
        k_neg = neg_targets.shape[1]
        for k in range(k_neg):
            assert torch.any(neg_targets[:, k, :] == target_batch) == False,\
                "all sampled negative targets must be different from positive targets"
    else:
        assert torch.any(neg_targets == target_batch) == False,\
                "all sampled negative targets must be different from positive targets"

def mask(loss, pad_mask):
    """
    loss    bs x (k_neg) x seq_len
    pad_mask   bs x seq_len
    k_neg_dim: whether k_neg dimension is in neg_targets
    """
    is_k_neg_dim = len(loss.shape) == 3
    if is_k_neg_dim:
        k_neg = loss.shape[1]
        for k in range(k_neg):
            loss[:, k, :].masked_fill_(pad_mask, 0.0)
    else:
        loss.masked_fill_(pad_mask, 0.0)
    return loss

def nll_loss(pos_logits, neg_logits, logits, ignore_index=None, reduce=True,
             org_targets = None,
             sample_version = 2,
             loss_version = 2):
    """_summary_

    Args:
        pos_logits: (bs, seq_len)
        neg_logits: (bs, k_neg, seq_len)
    
        # targets: (bs, neg_dim, seq_len)
        # neg_targets: (bs, neg_dim, seq_len)
        logits: current logits is (bs, k_neg, max_seq_len)
            we need to make it simple as (bs, max_seq_len)
            also, we want to offload some tensors from gpu that are for stats.
        ignore_index: _description_. Defaults to None.
        reduce: _description_. Defaults to True.
        org_targets: original targets from dataset. It preserves the batch size. shape: bs  x seq_len
        sampling_version: sampling version, it decides how to determine the number of negative samples.

    Returns:
        _type_: _description_
    """
    _pos_logits, neg_logits = torch.broadcast_tensors(pos_logits.unsqueeze(1), neg_logits)
    mu = torch.exp(neg_logits - _pos_logits)
    loss = -torch.log(1/(1 + mu.sum(dim=1, keepdim=True))) # 1 - 1/(1 + sum(e^{neg_i-pos}))
    loss = loss.squeeze(1)

    

    debug_training_sped = True
    if debug_training_sped:
        ignore_index = None
    if ignore_index is not None:
        # original targets are used to compute mask
        pad_mask = org_targets.eq(ignore_index)
        loss = loss.squeeze(-1)
        pad_mask = pad_mask.squeeze(-1)
        # RuntimeError: expand(CUDABoolType{[1, 457]}, size=[457]): the number of sizes provided (1) must be greater or equal to the number of dimensions in the tensor (2)
        loss = mask(loss, pad_mask)
    else:
        loss = loss.squeeze(-1)


    nce_log = {}


    if reduce:
        loss = loss.sum()
    
    if debug_training_sped:
        check_classification_acc = False
    else:
        check_classification_acc = True
    if check_classification_acc:
        max_neg_logits = torch.max(neg_logits, dim =1).values # hardest
        
        # org_pos_logits= logits.gather(dim=-1,index=org_targets.unsqueeze(-1)).squeeze(-1)
        if sample_version == 1:
            num_correct_classification = torch.sum((pos_logits > max_neg_logits).masked_fill_(pad_mask, False))
            # num_correct_classification = torch.sum((org_pos_logits > max_neg_logits).masked_fill_(pad_mask, False))
            num_tokens = torch.numel(org_targets) - torch.sum(pad_mask)
        else:
            num_correct_classification = torch.sum((org_pos_logits > max_neg_logits).masked_fill_(pad_mask, False))
            num_tokens = torch.numel(org_targets) - torch.sum(pad_mask) # exclude padding
        nce_log["class_acc"] = (num_correct_classification/num_tokens).item()
    return loss, nce_log

def seq_sampling(target_seq, vocab_size, k_neg = 1, validate_sampling=True):
    """
    sampling net targets for a single target sequence
    
    returns shape: 
    """
    assert len(target_seq.shape) == 1, "must be a sequence"
    # TODO: get statistics from dataset rather than data batch
    # neg sample size:  num_neg_sample * bs_size * length
    flattened_targets_l = torch.flatten(target_seq).tolist()
    
    # pos_attack = set([ 2,  5,  4,  6,  7, 32, 22,  8, 13, 68, 55, 26, 11, 56, 81, 63, 36, 21,
    #       9, 44]) # make them negative (remove from positive)
    # flattened_targets_l += pos_attack
    pos_token_ids_set = set(flattened_targets_l) 
    # pos_token_ids_set = set(flattened_targets_l) - pos_attack
    
    # get set all tokens ids and count the token ids
    # fill col of sample ids and excluding pos token ids

    num_diff_tokens = len(pos_token_ids_set)
    weights = torch.empty(vocab_size).fill_(
        1/(vocab_size - num_diff_tokens)
    )
    pos_token_indices = torch.tensor(list(pos_token_ids_set))
    # first dimension, positive token indices, zero weights
    weights = weights.index_fill(0, pos_token_indices, 0)
    device = target_seq.device
    max_seq_len = len(target_seq)
    sampled_neg_targets = torch.multinomial(weights, k_neg* max_seq_len, replacement=True).reshape((k_neg, max_seq_len)).to(device)
    
    validate_sampling = True
    target_seq = target_seq.unsqueeze(0).repeat((k_neg,1))
    if validate_sampling:
        assert torch.any(sampled_neg_targets == target_seq) == False,\
            "all sampled negative targets must be different from positive targets"

    # make them batch like
    return target_seq, sampled_neg_targets


def batch_sampling_v2(target_batch, vocab_size, k_neg=1, validate_sampling=True):
    """
    sampling net targets for the whole tgt length.  (seq_sampling)
    
    Return shape:
        (batch_size, k_neg, max_seq_length)
    """
    new_target_batch = []
    neg_target_batch = []
    for tgt_seq in target_batch: # along bs dimension
        tgt_seq, neg_seq = seq_sampling(tgt_seq, vocab_size, k_neg, validate_sampling=validate_sampling)
        # tgt_seq shape: (k_neg, max_seq_len)
        new_target_batch.append(tgt_seq) # NOTE: tho it can be optimized, as they are just single indices and k_neg won't be too large. The memory complexity is just O(k_neg * targets numel size * 1) = O(1).
        neg_target_batch.append(neg_seq)
    # new_target_batch shape: (batch_size * k_neg, max_seq_len)

    # stack to add a new dimension
    new_target_batch = torch.stack(new_target_batch) # TODO: improve efficiency
    neg_target_batch = torch.stack(neg_target_batch)
    return new_target_batch, neg_target_batch


def batch_sampling_hard_neg(target_batch, logits, k_neg=1, validate_sampling=True):
    """
    Based on model logits for all vocabulary terms, pointwise add up logits for
    all of token logits. Use them as the weights to sample negative samples.
    """
    flattened_targets = torch.flatten(target_batch)
    dim = len(logits.shape)
    weights = logits.sum(dim=range(dim-1)) # sum along the last dimension
    weights.index_fill_(0, flattened_targets, 0)
    bs, max_seq_len = target_batch.shape
    replacement = True
    sampled_neg_targets = torch.multinomial(weights, bs*k_neg*max_seq_len, replacement=replacement).reshape(bs, k_neg, max_seq_len)
    return sampled_neg_targets
    


def batch_sampling(target_batch, vocab_size, k_neg=1, validate_sampling=True):
    """
    sampling net targets for the whole tgt length
    """
    # TODO: get statistics from dataset rather than data batch
    # neg sample size:  num_neg_sample * bs_size * length
    flattened_targets = torch.flatten(target_batch)
    # pos_token_ids_set = set(flattened_targets.tolist())

    # get set all tokens ids and count the token ids
    # fill col of sample ids and excluding pos token ids
    
    with torch.no_grad():
        
        # weights = torch.ones(vocab_size).to(target_batch.device)
        weights = torch.tensor(list(de_freq_dict.values()), dtype=torch.float32).to(target_batch.device)

        # flattened_targets = flattened_targets.type(torch.cuda.LongTensor)
        weights.index_fill_(0, flattened_targets, 0)
        bs, max_seq_len = target_batch.shape
        replacement = True
        sampled_neg_targets = torch.multinomial(weights, bs*k_neg*max_seq_len, replacement=replacement).reshape(bs, k_neg, max_seq_len)

    if validate_sampling:
        validate_samples(sampled_neg_targets, target_batch)
    return sampled_neg_targets

def batch_sampling_v3(target_batch, vocab_size, k_neg=1, validate_sampling=True):
    """
    sampling net targets along the each seq length.

    It's deprecated because this sampling will introduce neighbour tokens which
    will want to make them dissimilar to the target tokens.
    """
    bs, seq_len = target_batch.shape
    batch_device = target_batch.device
    sampled_neg_targets = []

    for j in range(seq_len):
        flattened_targets = torch.flatten(target_batch[:,j])
        pos_token_ids_set = set(flattened_targets.tolist())
        num_diff_tokens = len(pos_token_ids_set)
        weights = torch.empty(vocab_size).fill_(
            1/(vocab_size - num_diff_tokens)
        )
        weights[flattened_targets] = 0
        sampled_neg_targets_j = torch.multinomial(weights, bs*k_neg, replacement=True).reshape(bs, k_neg)
        sampled_neg_targets.append(sampled_neg_targets_j)
        
    # import pdb; pdb.set_trace()
    # print('stack neg targets')
    # stack over seq len dimension
    sampled_neg_targets = torch.stack(sampled_neg_targets, dim=2)
    
    # reshape: bs x k_neg x seq_len -> bs x (k_neg * seq_len)
    # sampled_neg_targets = sampled_neg_targets.reshape(bs, k_neg,  seq_len).to(batch_device)
    sampled_neg_targets = sampled_neg_targets.to(batch_device)
    if validate_sampling:
        validate_samples(sampled_neg_targets, target_batch)
    return sampled_neg_targets


def batch_sampling_v4(target_batch, vocab_size, k_neg=1, validate_sampling=True):
    """
    Sampling neg targets by add target batch by a ramdom matrix and mod vocab size.

    It's deprecated because this sampling will introduce neighbour tokens which
    will want to make them dissimilar to the target tokens.

    Args:
        target_batch: _description_
        vocab_size: _description_
        k_neg: _description_. Defaults to 1.
        validate_sampling: _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    batch_device = target_batch.device
    def sample_random_tensor(l, h, shape):
        return torch.randint(low=l, high=h, size=shape)
    sampled_neg_targets = []
    for _ in range(k_neg):
        rand_additive = sample_random_tensor(1, vocab_size-1, target_batch.shape).to(batch_device)
        snt = torch.remainder((target_batch + rand_additive), vocab_size)
        sampled_neg_targets.append(snt)
        # validate_samples(snt, target_batch)
    sampled_neg_targets = torch.stack(sampled_neg_targets, dim=1)
    return sampled_neg_targets



@register_criterion("nce", dataclass=NCECriterionConfig)
class NoiseContrastiveEstimationCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.loss_version = 2
        self.k_neg = 200 # vocab size 35664
        self.sample_version = 1
        self.projection_layer = torch.nn.Linear(512, self.k_neg + 1)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # control hyperparameters here instead of command line
        
        net_output = model(**sample["net_input"], features_only=True)
        torch.cuda.empty_cache() 
        # import pdb; pdb.set_trace()
        # print('before nce')  # 3.6, 4.5, 8.1
        loss, _, nce_log= self.compute_loss(model, net_output, sample,
                            reduce=reduce,
                            loss_version=self.loss_version,
                            sample_version=self.sample_version,
                            k_neg = self.k_neg)
        # import pdb; pdb.set_trace()
        # print('after nce') # 5.6,  6.4, 8.1
        torch.cuda.empty_cache() # 4.6, 5.4, 5.3
        
        
        ce_loss = self.compute_ce_loss(model, net_output, sample, reduce=True)
        # ce_loss = 0
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # TODO
        # move loss version, sample version into 
        logging_output = {
            "loss": loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "k_neg":self.k_neg,
            "loss_version": self.loss_version,
            "sample_version": self.sample_version,
            "ce_loss": ce_loss,
        }
        if "class_acc" in nce_log:
            logging_output["class_acc"] = nce_log["class_acc"]
        return loss, sample_size, logging_output

    def compute_ce_loss(self, model, net_output, sample, reduce=True):
        model.eval()
        with torch.no_grad():
            hidden2vocab = net_output[2].T
            new_logits = net_output[0].matmul(hidden2vocab)
            net_output = [new_logits, net_output[1]]
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output).view(-1)
            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
        model.train()
        return loss


    def compute_loss(self, model, net_output, sample,
                    loss_version = 2,
                    sample_version = 2,
                    k_neg = 30,
                    reduce=True):
        
        targets = sample['target']
        logits = net_output[0]
        
        if len(net_output) == 2:
            vocab_size = net_output[0].size(2)
        else:
            # get vocab size from projection layer weights
            vocab2hidden = net_output[2]
            vocab_size, hidden_size = vocab2hidden.shape


        if sample_version == 1:
            neg_targets = batch_sampling(target_batch=targets, vocab_size=vocab_size, k_neg=k_neg)
            target_shape = targets.shape
            neg_targets_shape = neg_targets.shape

            
            pos_hidden2emb = F.embedding(targets, vocab2hidden)
            # pos_hidden2emb = vocab2hidden.index_select(0, targets.flatten()).reshape(*target_shape, hidden_size)

            neg_hidden2emb = F.embedding(neg_targets, vocab2hidden)
            # neg_hidden2emb = vocab2hidden.index_select(0, neg_targets.flatten()).reshape(*neg_targets_shape, hidden_size)
            empty_cache = False
            # import pdb; pdb.set_trace()
            # print('dd')

            pos_logits = torch.sum(logits * pos_hidden2emb, -1) # 0.3G

            if empty_cache:
                torch.cuda.empty_cache()

            # NOTE: in-place broadcast
            # logits, neg_hidden2emb = torch.broadcast_tensors(logits.unsqueeze(1), neg_hidden2emb)
            # neg_logits = torch.sum(logits * neg_hidden2emb, -1) # 0.5G
            neg_logits = torch.sum(logits.unsqueeze(1) * neg_hidden2emb, -1) # 1G
            if empty_cache:
                torch.cuda.empty_cache()
            loss, nce_log = nll_loss(
                pos_logits,
                neg_logits,
                logits = logits,
                ignore_index=self.padding_idx,
                reduce=reduce,
                org_targets = targets,
                sample_version = sample_version,
                loss_version = loss_version,
            )

        elif sample_version == 2:
            
            
            dup_targets, neg_targets = batch_sampling_v2(target_batch=targets, vocab_size=vocab_size, k_neg=k_neg)
            
            
            # logits = logits.unsqueeze(1).repeat_interleave(k_neg, dim=1) # TODO: it takes much more memory as k_neg grows 
            # logits = logits.repeat_interleave(k_neg, dim=0)
            loss, nce_log = nll_loss(
                # dup_targets,
                dup_targets,
                neg_targets,
                logits = logits,
                ignore_index=self.padding_idx,
                reduce=reduce,
                org_targets = targets,
                sample_version = sample_version,
                loss_version = loss_version,
            )
        elif sample_version == 3:
            neg_targets = batch_sampling_v3(target_batch=targets, vocab_size=vocab_size, k_neg=k_neg)
            # neg_targets = neg_targets.unsqueeze(0)
            # targets = targets.unsqueeze(0)
            loss, nce_log = nll_loss(
                targets,
                neg_targets,
                logits = logits,
                ignore_index=self.padding_idx,
                reduce=reduce,
                org_targets = targets,
                sample_version = sample_version,
                loss_version = loss_version,
            )
        elif sample_version == 4:
            neg_targets = batch_sampling_v4(target_batch=targets, vocab_size=vocab_size, k_neg=k_neg)
            # neg_targets = neg_targets.unsqueeze(0)
            # targets = targets.unsqueeze(0)
            loss, nce_log = nll_loss(
                targets,
                neg_targets,
                logits = logits,
                ignore_index=self.padding_idx,
                reduce=reduce,
                org_targets = targets,
                sample_version = sample_version,
                loss_version = loss_version,
            )
            
        elif sample_version == 5:
            neg_targets = batch_sampling(target_batch=targets, vocab_size=vocab_size, k_neg=k_neg)

            loss, nce_log = nll_loss(
                targets,
                neg_targets,
                logits = logits,
                ignore_index=self.padding_idx,
                reduce=reduce,
                org_targets = targets,
                sample_version = sample_version,
                loss_version = loss_version,
            )

        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))

        
        # target = model.get_targets(sample, net_output).view(-1)
        # loss = F.nll_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduction="sum" if reduce else "none",
        # )
        # print("NCE loss: ", loss)
        
        return loss, loss, nce_log

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        pred_acc = sum(log.get("class_acc", 0) for log in logging_outputs) / len(logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        metrics.log_scalar("class_acc", pred_acc, sample_size, round=3)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
