# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch

@dataclass
class NCECriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


def nll_loss(targets, neg_targets, logits, ignore_index=None, reduce=True,
             org_targets = None,
             version = 2):
    # if target.dim() == lprobs.dim() - 1:
    #     target = target.unsqueeze(-1)

    if org_targets is None:
        org_targets = targets

    
    x1 = logits.gather(dim=-1, index=torch.unsqueeze(targets, -1))
    x2 = logits.gather(dim=-1, index=torch.unsqueeze(neg_targets, -1))
    # import pdb; pdb.set_trace()
    # print('check gather')
    if version == 1:
        loss = -torch.log(1/(1 + torch.exp(x2-x1)))
    else:
        mu = torch.exp(x2-x1)
        loss = -torch.log(1/(1 + mu.sum(dim=1, keepdim=True)))
        loss = loss.squeeze(1) # squeeze k_neg dimension


    if ignore_index is not None:
        # original targets are used to compute mask
        pad_mask = org_targets.eq(ignore_index)
        loss = loss.squeeze(-1)
        loss.masked_fill_(pad_mask, 0.0)
    else:
        loss = loss.squeeze(-1)

    if reduce:
        # loss = loss.mean(dim=1)
        loss = loss.sum()
    return loss

def seq_sampling(target_seq, vocab_size, k_neg = 1, validate_sampling=True):
    """
    sampling net targets for a single target sequence
    
    returns shape: 
    """
    assert len(target_seq.shape) == 1, "must be a sequence"
    # TODO: get statistics from dataset rather than data batch
    # neg sample size:  num_neg_sample * bs_size * length
    flattened_targets = torch.flatten(target_seq)
    token_ids_set = set(flattened_targets.tolist())

    # get set all tokens ids and count the token ids
    # fill col of sample ids and excluding pos token ids

    num_diff_tokens = len(token_ids_set)
    weights = torch.empty(vocab_size).fill_(
        1/(vocab_size - num_diff_tokens)
    )
    pos_token_indices = torch.tensor(flattened_targets.tolist())
    # first dimension, positive token indices, zero weights
    weights.index_fill(0, pos_token_indices, 0) 
    device = target_seq.device
    max_seq_len = len(target_seq)
    sampled_neg_targets = torch.multinomial(weights, k_neg* max_seq_len, replacement=True).reshape((k_neg, max_seq_len)).to(device)
    
    target_seq = target_seq.unsqueeze(0).repeat((k_neg,1))
    if validate_sampling:
        assert torch.any(sampled_neg_targets == target_seq) == False,\
            "all sampled negative targets must be different from positive targets"

    # make them batch like
    return target_seq, sampled_neg_targets


def batch_sampling_v2(target_batch, vocab_size, k_neg=1, validate_sampling=True):
    """
    sampling net targets for the whole tgt length
    
    Return shape:
        (batch_size * k_neg, max_seq_length)
    """
    new_target_batch = []
    neg_target_batch = []
    for tgt_seq in target_batch:
        tgt_seq, neg_seq = seq_sampling(tgt_seq, vocab_size, k_neg, validate_sampling=True)
        # tgt_seq shape: (k_neg, max_seq_len)
        new_target_batch.append(tgt_seq)
        neg_target_batch.append(neg_seq)
    # new_target_batch shape: (batch_size * k_neg, max_seq_len)

    # stack to add a new dimension
    new_target_batch = torch.stack(new_target_batch) # TODO: improve efficiency
    neg_target_batch = torch.stack(neg_target_batch)
    return new_target_batch, neg_target_batch


def batch_sampling(target_batch, vocab_size, k_neg=1, validate_sampling=True):
    """
    sampling net targets for the whole tgt length
    """
    # TODO: get statistics from dataset rather than data batch
    # neg sample size:  num_neg_sample * bs_size * length
    flattened_targets = torch.flatten(target_batch)
    token_ids_set = set(flattened_targets.tolist())

    # get set all tokens ids and count the token ids
    # fill col of sample ids and excluding pos token ids

    num_diff_tokens = len(token_ids_set)
    weights = torch.empty(vocab_size).fill_(
        1/(vocab_size - num_diff_tokens)
    )
    for pos_token_id in flattened_targets.tolist():
        weights[pos_token_id] = 0
    batch_device = target_batch.device
    bs, max_seq_len = target_batch.shape
    sampled_neg_targets = torch.multinomial(weights, bs*k_neg*max_seq_len, replacement=True).reshape(bs, k_neg* max_seq_len).to(batch_device)
    if validate_sampling:
        assert torch.any(sampled_neg_targets == target_batch) == False,\
            "all sampled negative targets must be different from positive targets"
    return sampled_neg_targets

@register_criterion("nce", dataclass=NCECriterionConfig)
class NoiseContrastiveEstimationCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        vocab_size = net_output[0].size(2)
        targets = sample['target']
        version = 2
        logits = net_output[0]
        bs, max_seq_len, vocab_size = logits.shape
        if version == 1:
            neg_targets = batch_sampling(target_batch=targets, vocab_size=vocab_size, k_neg=1)
            loss = nll_loss(
                targets,
                neg_targets,
                logits = logits,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
        elif version == 2:
            k_neg = 3
            dup_targets, neg_targets = batch_sampling_v2(target_batch=targets, vocab_size=vocab_size, k_neg=k_neg)
            
            print('repeat interleave along a new dimension')
            logits = logits.unsqueeze(1).repeat_interleave(k_neg, dim=1)
            # logits = logits.repeat_interleave(k_neg, dim=0)
            

            loss = nll_loss(
                dup_targets,
                neg_targets,
                logits = logits,
                ignore_index=self.padding_idx,
                reduce=reduce,
                org_targets = targets
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
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
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
