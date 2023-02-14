import torch
import torch.nn as nn
from torch.nn import functional as F

def dynamic_kd_loss(student_logits, teacher_logits, temperature=1.0):

    with torch.no_grad():
        student_probs = F.softmax(student_logits, dim=-1)
        student_entropy = - torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1) # student entropy, (bsz, )
        # normalized entropy score by student uncertainty:
        # i.e.,  entropy / entropy_upper_bound
        # higher uncertainty indicates the student is more confusing about this instance
        instance_weight = student_entropy / torch.log(torch.ones_like(student_entropy) * student_logits.size(1))

    input = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits / temperature, dim=-1)
    batch_loss = F.kl_div(input, target, reduction="none").sum(-1) * temperature ** 2  # bsz
    weighted_kld = torch.mean(batch_loss * instance_weight)

    return weighted_kld
    