import os, argparse
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim

from model import base_model
from train.train_baseline import run
from utilities import config, utils, dataset as data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--output', type=str, default='baseline')
    parser.add_argument('--resume', action='store_true', help='resumed flag')
    parser.add_argument('--test', dest='test_only', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--gpu', default='0', help='the chosen gpu id')
    args = parser.parse_args()
    return args


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("seed: ", seed)


def saved_for_eval(eval_loader, result, output_path, epoch=None):
    """
        save a results file in the format accepted by the submission server.
        input result : [ans_idxs, acc, q_ids]
    """
    label2ans = eval_loader.dataset.label2ans
    results = []
    for a, q_id in zip(result[0], result[2]):
        results.append({'question_id': int(q_id), 'answer': label2ans[a]})
    results_path = os.path.join(output_path, 'eval_results.json')
    with open(results_path, 'w') as fd:
        json.dump(results, fd)


if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.test_only:
        args.resume = True
    output_path = 'saved_models/{}/{}'.format(config.type + config.version, args.output)
    utils.create_dir(output_path)
    torch.backends.cudnn.benchmark = True

    ######################################### DATASET PREPARATION #######################################
    if config.mode == 'train':
        train_loader = data.get_loader('train')
        val_loader = data.get_loader('val')
    elif args.test_only:
        train_loader = None
        val_loader = data.get_loader('test')
    else:
        train_loader = data.get_loader('trainval')
        val_loader = data.get_loader('test')

    ######################################### MODEL PREPARATION #######################################
    embeddings = np.load(os.path.join(config.cache_root, 'glove6b_init_300d.npy'))
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(embeddings, val_loader.dataset.num_ans_candidates).cuda()
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adamax(model.parameters())

    r = np.zeros(3)
    start_epoch = 0
    acc_val_best = 0
    tracker = utils.Tracker()
    model_path = os.path.join(output_path, 'model.pth')
    if args.resume:
        model_path = os.path.join(output_path, 'model.pth')
        logs = torch.load(model_path)
        start_epoch = logs['epoch']
        acc_val_best = logs['acc_val_best']
        model.load_state_dict(logs['model_state'])
        optimizer.load_state_dict(logs['optim_state'])

    ######################################### MODEL RUN #######################################
    for epoch in range(start_epoch, config.epochs):
        if not args.test_only:
            run(model, train_loader, optimizer, tracker, train=True, prefix='train', epoch=epoch)
        if not (config.mode == 'trainval' and epoch in range(config.epochs - 5)):
            r = run(model, val_loader, optimizer, tracker, train=False,
                      prefix='val', epoch=epoch, has_answers=(config.mode == 'train'))

        if not args.test_only:
            results = {
                'epoch': epoch,
                'acc_val_best': acc_val_best,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
            }
            if config.mode == 'train' and r[1].mean() > acc_val_best:
                acc_val_best = r[1].mean()
                saved_for_eval(val_loader, r, output_path, epoch)
            if config.mode == 'trainval':
                if epoch in range(config.epochs - 5, config.epochs):
                    saved_for_eval(val_loader, r, output_path, epoch)
                    torch.save(results, model_path+'{}.pth'.format(epoch))
            torch.save(results, model_path)
        else:
            saved_for_eval(val_loader, r, output_path, epoch)
            break

