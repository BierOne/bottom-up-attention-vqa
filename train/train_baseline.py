import torch
import torch.nn as nn
from tqdm import tqdm


def binary_cross_entropy_with_logits(input, target, mean=False):
    """
    Function that measures Binary Cross Entropy between target and output logits:
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    loss = loss.sum(dim=1)
    return loss.mean() if mean else loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros_like(labels).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores.sum(dim=1)


def run(model, loader, optimizer, tracker, train=False, has_answers=True, prefix='', epoch=0):
    if train:
        model.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        model.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ, q_ids, accs = [], [], []
    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))

    for i, (v, b, q, a, qid) in enumerate(loader):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()
        pred, _ = model(v, b, q, a)
        if has_answers:
            loss = binary_cross_entropy_with_logits(pred, a, True)
            acc = compute_score_with_logits(pred, a.data)
        if train:
            # optimze parameters
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()
        else:
            # store information about evaluation of this minibatch
            answer_idx = pred.max(dim=1)[1].cpu()
            answ.append(answer_idx.view(-1))
            q_ids.append(qid.view(-1))
            if has_answers:
                accs.append(acc.view(-1).cpu())

        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value),
                               acc=fmt(acc_tracker.mean.value),
                               )
    if not train:
        answ = torch.cat(answ, dim=0).numpy()
        q_ids = torch.cat(q_ids, dim=0).numpy()
        if has_answers:
            accs = torch.cat(accs, dim=0).cpu().numpy()
        return answ, accs, q_ids
