# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from typing import Optional
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from einops import rearrange

@dataclass
class KDLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    regressor: bool = field(
        default=False,
        metadata={"help": "loss type mse or kld"},
    )
    loss_type: str = field(
        default='mse',
        metadata={"help": "loss type mse or kld"},
    )
    decoder_kd: bool = field(
        default=False, metadata={"help": "decoder attention distillation"}
    )
    self_kd: bool = field(
        default=True, metadata={"help": "decoder attention distillation"}
    )
    cross_kd: bool = field(
        default=True, metadata={"help": "decoder attention distillation"}
    )
    value_kd: bool = field(
        default=False, metadata={"help": "value relation distillation"}
    )
    rambda: int = field(
        default=1000000,
        metadata={"help": "attn_loss weight"},
    )
    decay: float = field(
        default=0.985,
        metadata={"help": "decay value for rambda"}
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    kd_rate: Optional[float] = field(
        default=None,
        metadata={"help": "the hyperparameter `tau` to control the number of words to get distillation knowledge"}
    )
    kd_queue_size: Optional[int] = field(
        default=20000, 
        metadata={"help": "queue size for global_level, batch_level and global_multi_level selection"}
    )
    student_temp: float = field(
        default=1,
        metadata={"help": "student model temperature for distillation"}
    )
    teacher_temp: float = field(
        default=1,
        metadata={"help": "teacher model emperature for distillation"}
    )
    alpha: Optional[float] = field(
        default=None,
        metadata={"help": "KD loss weightage, 0 means pure training without KD"}
    )
    use_adaptive_weightage: bool = field(
        default=False,
        metadata={"help": "whether to use adaptive weightage for loss terms during KD"}
    )
    adaptive_smoothing: Optional[float] = field(
        default=None,
        metadata={"help": "beta for smoothing factor in the sigmoid function"}
    )
    use_adaptive_kd_rates: bool = field(
        default=False,
        metadata={"help": "whether to use adaptive distil rate, i.e. different distil rates for different languages"}
    )
    kd_selection_temp: Optional[float] = field(
        default=None,
        metadata={"help": "temperature value for generating distil rates"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "kd_label_smoothed_cross_entropy", dataclass=KDLabelSmoothedCrossEntropyCriterionConfig
)
class KDLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        kd_rate,
        kd_queue_size,
        student_temp,
        teacher_temp,
        alpha,
        use_adaptive_weightage,
        adaptive_smoothing,
        use_adaptive_kd_rates,
        kd_selection_temp,
        rambda,
        decay,
        loss_type,
        decoder_kd,
        self_kd,
        cross_kd,
        value_kd,
        regressor,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        # new parameters
        self.kd_strategy = self.task.kd_strategy
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.kd_rate = kd_rate
        self.kd_queue_size = kd_queue_size
        self.num_languages = len(self.task.src_lang_ids)
        self.use_adaptive_weightage = use_adaptive_weightage
        self.use_adaptive_kd_rates = use_adaptive_kd_rates
        self.kd_selection_temp = kd_selection_temp
        self.alpha = alpha if not use_adaptive_weightage else None
        self.beta = 1 if adaptive_smoothing is not None else adaptive_smoothing
        self.rambda = rambda
        self.decay = decay
        self.loss_type = loss_type
        self.decoder_kd = decoder_kd
        self.self_kd = self_kd
        self.cross_kd = cross_kd
        self.value_kd = value_kd
        self.regressor = regressor
        if self.kd_strategy == "global_multi_level":
            self.queue = {}
            for id in self.task.src_lang_ids:
                self.queue[id] = torch.cuda.FloatTensor([])
        else:
            self.queue = torch.cuda.FloatTensor([])

    
    def get_lang_kd_rates(self, indices, T=1):
        if self.use_adaptive_kd_rates:
            lens = torch.cuda.FloatTensor([len(v) for v in indices.values()])
            lens_prob = F.softmax((1/lens)/T, dim=-1, dtype=torch.float32).tolist()
            return lens_prob
        else:
            return [self.kd_rate] * len(indices)


    def get_lang_ids(self, tokens):
        non_pad_mask = tokens.ne(self.padding_idx)
        col_indices = torch.max(non_pad_mask, dim=1)[1]
        col_indices = col_indices.unsqueeze(1)
        lang_ids = tokens.gather(1, col_indices)
        return lang_ids.flatten().tolist()


    def push_to_FIFO_queue(self, tensor):
        # this method is applicable only when we have a single global queue
        # here self.queue is torch.cuda.FloatTensor
        tensor = tensor.detach()
        tensor_sz = tensor.size(0)
        current_queue_sz = self.queue.size(0)
        if tensor_sz + current_queue_sz >= self.kd_queue_size:
            self.queue = self.queue[tensor_sz: ]
        self.queue = torch.cat((self.queue, tensor))


    def push_to_lang_FIFO_queue(self, id, tensor):
        # this method is applicable only when we have a mulitple global queues
        # here self.queue is dictionary of torch.cuda.FloatTensors
        tensor = tensor.detach()
        tensor_sz = tensor.size(0)
        current_queue_sz = self.queue[id].size(0)
        if tensor_sz + current_queue_sz >= self.kd_queue_size:
            self.queue[id] = self.queue[id][tensor_sz: ]
        self.queue[id] = torch.cat((self.queue[id], tensor))


    def forward(self, model, sample, epoch=None, reduce=True, teacher_maps=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, (attn_output, value_relation, regressed_maps) = model(**sample["net_input"])
        # print(regressed_maps)
        attn_output = attn_output
        decoder_self_attn_output = net_output[1]['self_attn_tensor']
        decoder_cross_attn_output = net_output[1]['cross_attn_tensor']
        teacher_output = sample.get("teacher_output", None)
        teacher_attn_output = sample.get("teacher_attn_output", None)
        teacher_decoder_self_attn_output = sample.get("teacher_decoder_self_attn_output", None)
        teacher_decoder_cross_attn_output = sample.get("teacher_decoder_cross_attn_output", None)
        teacher_value_relation = sample.get("teacher_value_relation", None)
        
        loss, extra = self.compute_loss(
            model, 
            net_output, 
            sample,
            epoch,
            teacher_output=teacher_output,
            attn=attn_output,
            decoder_self_attn=decoder_self_attn_output,
            decoder_cross_attn=decoder_cross_attn_output,
            teacher_attn=teacher_attn_output,
            teacher_decoder_self_attn=teacher_decoder_self_attn_output,
            teacher_decoder_cross_attn=teacher_decoder_cross_attn_output,
            value_relation = value_relation,
            teacher_value_relation = teacher_value_relation,
            regressed_maps = regressed_maps
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'kd_loss': extra['kd_loss'].data if extra.get('kd_loss', None) is not None else 0,
            'nll_loss_student': extra['nll_loss_student'].data if extra.get('nll_loss_student', None) is not None else loss.data,
            'nll_loss_teacher': extra['nll_loss_teacher'].data if extra.get('nll_loss_teacher', None) is not None else 0,
            'attn_loss' : extra['attn_loss'].data if extra.get('attn_loss', None) is not None else 0,
            'decoder_self_attn_loss': extra['decoder_self_attn_loss'].data if extra.get('decoder_self_attn_loss', None) is not None else 0,
            'decoder_cross_attn_loss': extra['decoder_cross_attn_loss'].data if extra.get('decoder_cross_attn_loss', None) is not None else 0,
            'rep_loss': extra['rep_loss'].data if extra.get('rep_loss', None) is not None else 0,
            'golden_loss': extra['golden_loss'].data if extra.get('golden_loss', None) is not None else 0,
            'weight': extra['weight'].data if extra.get('weight', None) is not None else 0,
            # 'value_relation_loss': extra['value_relation_loss'].data if extra.get('value_relation_loss', None) is not None else 0,
            # 'regression_loss': extra['regression_loss'].data if extra.get('regression_loss', None) is not None else 0
        }
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    def compute_loss(self, model, net_output, sample, epoch=None, teacher_output=None, attn=None, decoder_self_attn=None, decoder_cross_attn=None, teacher_attn=None, teacher_decoder_self_attn=None, teacher_decoder_cross_attn=None, value_relation=None, teacher_value_relation=None, regressed_maps=None):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        pad_mask = target.eq(self.padding_idx).view(-1)
        KD_mask = None
        extra = dict()

        # get student logits
        student_logits = net_output[0]
        student_logits = student_logits.view(-1, student_logits.size(-1))
        student_logits_T = student_logits/self.student_temp

        # get teacher probs
        teacher_logits = teacher_output[0]
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        teacher_probs_T = F.softmax(teacher_logits/self.teacher_temp, dim=-1, dtype=torch.float32)

        # compute teacher log-probs to get teacher loss value
        teacher_lprobs = sample.get("teacher_lprobs", None)

        # compute preliminary loss and nll_loss of student_model
        golden_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, 
            target, 
            self.eps, 
            ignore_index=self.padding_idx, 
            reduce=False
        )

        if teacher_lprobs is not None:
            # compute preliminary lprobs, loss, nll_loss of teacher_model
            teacher_lprobs = teacher_lprobs.view(-1, teacher_lprobs.size(-1))
            _, nll_loss_teacher = label_smoothed_nll_loss(
                teacher_lprobs, 
                target, 
                self.eps, 
                ignore_index=self.padding_idx, 
                reduce=False
            )

        nll_loss = nll_loss.view(-1)
        nll_loss_teacher = nll_loss_teacher.view(-1)
        golden_loss = golden_loss.view(-1)
        extra['golden_loss'] = golden_loss.sum()

        if teacher_output is None:
            loss = golden_loss
            
        elif self.kd_strategy == 'word_and_seq_level':
            kd_loss = F.cross_entropy(
                student_logits_T,
                teacher_probs_T,
                reduction='none'
            )
            kd_loss.masked_fill_(pad_mask, 0)
            extra['kd_loss'] = kd_loss.sum()
            extra['nll_loss_student'] = nll_loss.sum()
            extra['nll_loss_teacher'] = nll_loss_teacher.sum()
            if self.use_adaptive_weightage:
                with torch.no_grad():
                    self.alpha = F.relu(torch.tanh(self.beta * (nll_loss_teacher - nll_loss)))
            loss = ((1.0 - self.alpha) * golden_loss).sum() + (self.alpha * kd_loss).sum()

        elif not self.use_adaptive_weightage and self.kd_strategy == 'batch_level':
            loss_gate = nll_loss.topk(
                math.ceil(
                    nll_loss.size(0) * self.kd_rate
                ), 
                dim=0, 
                largest=True
            )[0][-1]
            KD_mask = nll_loss < loss_gate
            kd_loss = F.cross_entropy(
                student_logits_T,
                teacher_probs_T,
                reduction='none'
            )
            kd_loss.masked_fill_(pad_mask, 0)
            kd_loss = kd_loss.view(-1)
            kd_loss = kd_loss[~KD_mask]
            extra['kd_loss'] = kd_loss.sum()
            extra['nll_loss_student'] = nll_loss.sum()
            extra['nll_loss_teacher'] = nll_loss_teacher.sum()
            loss = (1.0 - self.alpha) * golden_loss.sum() + self.alpha * kd_loss.sum()

        elif not self.use_adaptive_weightage and self.kd_strategy == 'global_level':
            kd_loss = F.cross_entropy(
                student_logits_T,
                teacher_probs_T,
                reduction='none'
            )
            # from the queue get the gate
            self.push_to_FIFO_queue(nll_loss)
            loss_gate = self.queue.topk(
                math.ceil(
                    self.queue.size(0) * self.kd_rate
                ), 
                dim=0, 
                largest=True
            )[0][-1]
            KD_mask = nll_loss < loss_gate # B * T
            kd_loss.masked_fill_(pad_mask, 0)
            kd_loss = kd_loss.view(-1)
            kd_loss = kd_loss[~KD_mask]
            extra['kd_loss'] = kd_loss.sum()
            extra['nll_loss_student'] = nll_loss.sum()
            extra['nll_loss_teacher'] = nll_loss_teacher.sum()
            loss = (1.0 - self.alpha) * golden_loss.sum() + self.alpha * kd_loss.sum()

        elif not self.use_adaptive_weightage and self.kd_strategy == "global_multi_level":
            # add language-wise losses to their respective queues
            kd_loss = F.cross_entropy(
                student_logits_T,
                teacher_probs_T,
                reduction='none'
            )
            kd_loss.masked_fill_(pad_mask, 0)
            kd_loss = kd_loss.view(-1)
            indices, total_kd_loss = dict(), 0
            inp_tokens = sample["net_input"]["src_tokens"]
            for idx, val in enumerate(self.get_lang_ids(inp_tokens)):
                indices.setdefault(val, []).append(idx)
            nll_loss = nll_loss.view(inp_tokens.size(0), -1)
            for key, val in indices.items():
                nll_loss_lang = nll_loss.index_select(0, torch.cuda.LongTensor(val)).view(-1)
                self.push_to_lang_FIFO_queue(key, nll_loss_lang)
            kd_rates = self.get_lang_kd_rates(indices, self.kd_selection_temp)
            
            for idx, kd_rate in zip(indices.keys(), kd_rates):
                loss_gate = self.queue[idx].topk(
                    math.ceil(
                        self.queue[idx].size(0) * kd_rate
                    ), 
                    dim=0, 
                    largest=True
                )[0][-1]
                KD_mask = nll_loss_lang >= loss_gate
                KD_indices = KD_mask.nonzero().view(-1)
                total_kd_loss += kd_loss.gather(0, KD_indices).sum()
            extra['kd_loss'] = total_kd_loss
            extra['nll_loss_student'] = nll_loss.sum()
            extra['nll_loss_teacher'] = nll_loss_teacher.sum()
            loss = (1.0 - self.alpha) * golden_loss.sum() + self.alpha * total_kd_loss

        else:
            raise ValueError("unknown strategy or parameter mismatch")
        attn_loss = None
        decoder_self_attn_loss = None
        decoder_cross_attn_loss = None
        rep_loss = None
        value_relation_loss = None
        regression_loss= None
        if epoch:
            if epoch <=100:
                if attn is not None and teacher_attn is not None and epoch is not None:
                    if self.loss_type == 'mse':
                        attn_loss = F.mse_loss((attn), teacher_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                        if self.value_kd:
                            x = value_relation.shape[2]
                            value_relation_loss = F.kl_div(F.log_softmax(rearrange(teacher_value_relation, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(value_relation, 'B C H W -> B C (H W)'), dim=-1), reduction='batchmean', log_target=True) * self.rambda * (self.decay ** (epoch-1)) / (x * 4)
                        if self.decoder_kd:
                            if self.self_kd and self.cross_kd:
                                decoder_self_attn_loss = F.mse_loss(decoder_self_attn, teacher_decoder_self_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1)) / 2
                                decoder_cross_attn_loss = F.mse_loss(decoder_cross_attn, teacher_decoder_cross_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1)) / 2
                                
                            elif self.self_kd:
                                decoder_self_attn_loss = F.mse_loss(decoder_self_attn, teacher_decoder_self_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                            elif self.cross_kd:
                                decoder_cross_attn_loss = F.mse_loss(decoder_cross_attn, teacher_decoder_cross_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                        
                    elif self.loss_type == 'mse_uncertainty':
                        attn_loss = F.mse_loss((attn), teacher_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                        if self.value_kd:
                            x = value_relation.shape[2]
                            value_relation_loss = F.kl_div(F.log_softmax(rearrange(teacher_value_relation, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(value_relation, 'B C H W -> B C (H W)'), dim=-1), reduction='batchmean', log_target=True) * self.rambda * (self.decay ** (epoch-1)) / (x * 4)
                        if self.decoder_kd:
                            if self.self_kd and self.cross_kd:
                                decoder_self_attn_loss = F.mse_loss(decoder_self_attn, teacher_decoder_self_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1)) / 2
                                decoder_cross_attn_loss = F.mse_loss(decoder_cross_attn, teacher_decoder_cross_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1)) / 2
                                
                            elif self.self_kd:
                                decoder_self_attn_loss = F.mse_loss(decoder_self_attn, teacher_decoder_self_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                            elif self.cross_kd:
                                decoder_cross_attn_loss = F.mse_loss(decoder_cross_attn, teacher_decoder_cross_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                        rep_loss = (attn_loss + decoder_self_attn_loss + decoder_cross_attn_loss).sum()
                        extra['attn_loss'] = attn_loss.sum()
                        extra['decoder_self_attn_loss'] = decoder_self_attn_loss.sum()
                        extra['decoder_cross_attn_loss'] = decoder_cross_attn_loss.sum()
                        loss = ((1.0 - self.alpha) * golden_loss).sum()

                        probs = torch.exp(lprobs)
                        entropy = torch.sum(probs * lprobs, dim=1)  # bsz
                        avg_prob = 1 / probs.shape[-1] * torch.ones((1, probs.shape[-1]))                      
                        # normalize the entropy to  0 to 1
                        weight = entropy / torch.sum(avg_prob * torch.log(avg_prob))  # bsz
                        weight_mean = torch.mean(weight, dim=0) # scalar
                        
                        rep_loss = weight_mean * rep_loss
                        kd_loss =  (1 - weight_mean) * self.alpha * kd_loss.sum()
                        extra['rep_loss'] = rep_loss
                        extra['kd_loss'] = kd_loss
                        loss += rep_loss + kd_loss
                        return loss, extra
                    
                    elif self.loss_type == 'regression':
                        regression_loss = F.mse_loss((attn), regressed_maps, reduction='mean') * self.rambda * (self.decay ** (epoch-1)) /50
                        if self.value_kd:
                            x = value_relation.shape[2]
                            value_relation_loss = F.kl_div(F.log_softmax(rearrange(teacher_value_relation, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(value_relation, 'B C H W -> B C (H W)'), dim=-1), reduction='batchmean', log_target=True) * self.rambda * (self.decay ** (epoch-1)) / (x * 4)
                        if self.decoder_kd:
                            decoder_attn_loss = F.mse_loss(decoder_attn, teacher_decoder_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                            
                    elif self.loss_type =='kld_uncertainty':
                        loss = ((1.0 - self.alpha) * golden_loss).sum()

                        attn_loss =F.kl_div(F.log_softmax(rearrange(attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.rambda
                        if self.decoder_kd:
                            if self.self_kd and self.cross_kd:
                                decoder_self_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) / 2 * self.rambda
                                decoder_cross_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) / 2 * self.rambda
                            elif self.self_kd:
                                decoder_self_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.rambda
                            elif self.cross_kd:
                                decoder_cross_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.rambda
                        rep_loss = attn_loss + decoder_self_attn_loss + decoder_cross_attn_loss
                        extra['attn_loss'] = attn_loss.sum()
                        extra['decoder_self_attn_loss'] = decoder_self_attn_loss.sum()
                        extra['decoder_cross_attn_loss'] = decoder_cross_attn_loss.sum()

                        probs = torch.exp(lprobs)
                        entropy = torch.sum(probs * lprobs, dim=1)  # bsz
                        avg_prob = 1 / probs.shape[-1] * torch.ones((1, probs.shape[-1]))                
                        weight = entropy / torch.sum(avg_prob * torch.log(avg_prob))  # bsz
                        weight_mean = torch.mean(weight, dim=0)
                        
                        loss = ((1.0 - self.alpha) * golden_loss).sum()
                        rep_loss = weight_mean * rep_loss * 2
                        kd_loss =  (1 - weight_mean) * self.alpha * kd_loss.sum() * 2
                        extra['weight'] = weight_mean
                        extra['rep_loss'] = rep_loss
                        extra['kd_loss'] = kd_loss
                        loss += rep_loss + kd_loss
                        return loss, extra

                    elif self.loss_type == 'kld':
                        attn_loss =F.kl_div(F.log_softmax(rearrange(attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.rambda
                        if self.decoder_kd:
                            if self.self_kd and self.cross_kd:
                                decoder_self_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) / 2 * self.rambda
                                decoder_cross_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) / 2 * self.rambda
                            elif self.self_kd:
                                decoder_self_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_self_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.rambda
                            elif self.cross_kd:
                                decoder_cross_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_cross_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='sum', log_target=True) * (self.decay ** (epoch-1)) * self.rambda
                        # if self.value_kd:
                            # x = value_relation.shape[2]
                            # value_relation_loss = F.kl_div(F.log_softmax(rearrange(teacher_value_relation, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(value_relation, 'B C H W -> B C (H W)'), dim=-1), reduction='batchmean', log_target=True) * self.rambda * (self.decay ** (epoch-1)) / (x * 4)  
                    elif self.loss_type == 'decoder':
                        if self.self_kd and self.cross_kd:
                            decoder_self_attn_loss = F.mse_loss(decoder_self_attn, teacher_decoder_self_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1)) / 2
                            decoder_cross_attn_loss = F.mse_loss(decoder_cross_attn, teacher_decoder_cross_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1)) / 2

                        elif self.self_kd:
                            decoder_self_attn_loss = F.mse_loss(decoder_self_attn, teacher_decoder_self_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                        elif self.cross_kd:
                            decoder_cross_attn_loss = F.mse_loss(decoder_cross_attn, teacher_decoder_cross_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1))
                    elif self.loss_type == 'minilm':
                        attn = attn[:,-4:, :, :]
                        teacher_attn = teacher_attn[:,-4:, :, :]
                        value_relation = value_relation[:, -4:, :, :]
                        teacher_value_relation = teacher_value_relation[:, -4:, :, :]
                        decoder_attn = decoder_attn[:, -4:, :, :]
                        teacher_decoder_attn = teacher_decoder_attn[:, -4:, :, :]
                        attn_loss = F.kl_div(F.log_softmax(rearrange(teacher_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(attn, 'B C H W -> B C (H W)'), dim=-1), reduction='batchmean', log_target=True) * self.rambda * (self.decay ** (epoch-1))
                        if self.value_kd:
                            x = value_relation.shape[2]
                            value_relation_loss = F.kl_div(F.log_softmax(rearrange(teacher_value_relation, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(value_relation, 'B C H W -> B C (H W)'), dim=-1), reduction='batchmean', log_target=True) * self.rambda * (self.decay ** (epoch-1)) / (x * 4)
                        if self.decoder_kd:
                            decoder_attn_loss = F.kl_div(F.log_softmax(rearrange(decoder_attn, 'B C H W -> B C (H W)'), dim=-1), F.log_softmax(rearrange(teacher_decoder_attn, 'B C H W -> B C (H W)'), dim=-1), reduction='batchmean', log_target=True) * self.rambda * (self.decay ** (epoch-1))

                    # elif self.loss_type == 'ckd':
                        
                    # if KD_mask is not None:
                    #     B, H, T, S = decoder_attn.shape
                    #     decoder_attn_loss = F.mse_loss(decoder_attn, teacher_decoder_attn, reduction='none') * self.rambda * (self.decay ** (epoch-1))
                    #     decoder_attn_loss = decoder_attn_loss.transpose(1,2).reshape(B*T,H,S)[~KD_mask].mean()
                    # else:
            else: 
                if attn is not None and teacher_attn is not None and epoch is not None:
                    attn_loss = F.kl_div(attn, teacher_attn, reduction='mean') * self.rambda * 0
                    value_relation_loss = F.kl_div(value_relation, teacher_value_relation, reduction='mean') * self.rambda * 0

                    # if KD_mask is not None:
                    #     B, H, T, S = decoder_attn.shape
                    #     decoder_attn_loss = F.mse_loss(decoder_attn, teacher_decoder_attn, reduction='none') * self.rambda * (self.decay ** (epoch-1))
                    #     decoder_attn_loss = decoder_attn_loss.transpose(1,2).reshape(B*T,H,S)[~KD_mask].mean()
                    # else:
                    # decoder_attn_loss = F.mse_loss(decoder_attn, teacher_decoder_attn, reduction='mean') * self.rambda * (self.decay ** (epoch-1)) * 0
        if attn_loss:
            extra['attn_loss'] = attn_loss.sum()
            loss += attn_loss
        if decoder_self_attn_loss:
            extra['decoder_self_attn_loss'] = decoder_self_attn_loss.sum()
            loss += decoder_self_attn_loss
        if decoder_cross_attn_loss:
            extra['decoder_cross_attn_loss'] = decoder_cross_attn_loss.sum()
            loss += decoder_cross_attn_loss
        if value_relation_loss:
            extra['value_relation_loss'] = value_relation_loss.sum()
            loss += value_relation_loss
        if regression_loss:
            extra['regression_loss'] = regression_loss.sum()
            loss += regression_loss
        return loss, extra


    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # sum metrics
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_student = sum(log.get('nll_loss_student', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nll_loss_teacher = sum(log.get('nll_loss_teacher', 0) for log in logging_outputs)
        kd_loss = sum(log.get('kd_loss', 0) for log in logging_outputs)
        attn_loss = sum(log.get('attn_loss', 0) for log in logging_outputs)
        # value_relation_loss = sum(log.get('value_relation_loss', 0) for log in logging_outputs)
        # regression_loss = sum(log.get('regression_loss', 0) for log in logging_outputs)
        decoder_self_attn_loss = sum(log.get('decoder_self_attn_loss', 0) for log in logging_outputs)
        decoder_cross_attn_loss = sum(log.get('decoder_cross_attn_loss', 0) for log in logging_outputs)
        rep_loss = sum(log.get('rep_loss', 0) for log in logging_outputs)
        golden_loss = sum(log.get('golden_loss', 0) for log in logging_outputs)
        weight = sum(log.get('weight', 0) for log in logging_outputs)
        # log metrics
        metrics.log_scalar(
            'loss', 
            loss / sample_size / math.log(2), 
            sample_size, 
            round=3
        )
        metrics.log_scalar(
            'attn_loss', 
            attn_loss / sample_size / math.log(2), 
            sample_size, 
            round=3
        )
        metrics.log_scalar(
            'rep_loss', 
            rep_loss / sample_size / math.log(2), 
            sample_size, 
            round=3
        )
        metrics.log_scalar(
            'golden_loss', 
            golden_loss / sample_size / math.log(2), 
            sample_size, 
            round=3
        )
        metrics.log_scalar(
            'decoder_self_attn_loss', 
            decoder_self_attn_loss / sample_size / math.log(2), 
            sample_size, 
            round=3
        )
        metrics.log_scalar(
            'decoder_cross_attn_loss', 
            decoder_cross_attn_loss / sample_size / math.log(2), 
            sample_size, 
            round=3
        )
        metrics.log_scalar(
            'nll_loss', 
            nll_loss_student / ntokens / math.log(2), 
            ntokens, 
            round=3)
        metrics.log_scalar(
            'nll_loss_teacher', 
            nll_loss_teacher / ntokens / math.log(2), 
            ntokens, 
            round=3)
        metrics.log_scalar(
            'kd_loss', 
            kd_loss / ntokens / math.log(2), 
            ntokens, 
            round=3)
        metrics.log_scalar(
            'weight', 
            weight / 16,
        round=3)
        metrics.log_derived(
            'ppl', 
            lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
