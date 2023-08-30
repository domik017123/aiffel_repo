from dataset.dataset import Im2LatexDataset
import os
import sys
import argparse
import logging
import yaml
from torchtext.data import metrics
import numpy as np
import torch
from munch import Munch
from tqdm.auto import tqdm
from src.models import get_model, Model
from src.utils import *
from eval import detokenize
import datetime as dt
from dateutil.tz import gettz
import warnings
warnings.filterwarnings('ignore')


def evaluate(model: Model, dataset: Im2LatexDataset, args: Munch, num_batches: int = None, name: str = 'test'):
    assert len(dataset) > 0
    device = torch.device(args.device)
    # log = {}
    # bleus = []
    losses = []
    # bleu_score = 0
    for i, (seq, im) in enumerate(iter(dataset)):
        if seq is None or im is None:
            continue
        tgt_seq, tgt_mask = seq['input_ids'].to(device), seq['attention_mask'].bool().to(device)
        encoded = model.encoder(im.to(device))
        loss = model.decoder(tgt_seq, mask=tgt_mask, context=encoded)
        losses.append(loss.item())
        # dec = model.decoder.generate(torch.LongTensor([args.bos_token]*len(encoded))[:, None].to(device), args.max_seq_len,
        #                              eos_token=args.pad_token, context=encoded, temperature=args.get('temperature', .2))
        # pred = detokenize(dec, dataset.tokenizer)
        # truth = detokenize(seq['input_ids'], dataset.tokenizer)
        # bleu = []
        # for i in range(len(pred)):
        #     bleu.append(metrics.bleu_score([pred[i]], [[truth[i]]], max_n=1, weights=[1]))    
        # bleus.append(np.mean(bleu))
        if num_batches is not None and i >= num_batches:
            break
    # if len(bleus) > 0:
    #     bleu_score = np.mean(bleus)
    #     log[name+'/bleu'] = bleu_score
    # if len(loss) > 0:
        # bleu_score = np.mean(bleus)
        # log[name+'/bleu'] = bleu_score
    print('VAL_LOSS: %.5f' % np.mean(losses))
    return np.mean(losses)

def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args, test=False)
    valdataloader = Im2LatexDataset().load(args.valdata)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)
    device = torch.device('cuda')
    model = get_model(args)
    #if args.load_chkpt is not None:
    #    model.load_state_dict(torch.load(args.load_chkpt, map_location=device))
    encoder, decoder = model.encoder, model.decoder

    def save_models(e):
        torch.save(model.state_dict(), os.path.join(args.out_path, '%s_e%02d.pth' % (args.name, e+1)))
        yaml.dump(dict(args), open(os.path.join(args.out_path, 'config.yaml'), 'w+'))

    opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas)

    for e in range(args.epoch, args.epochs):
        args.epoch = e
        dset = tqdm(iter(dataloader))
        for i, (seq, im) in enumerate(dset):
            if seq is not None and im is not None:
                opt.zero_grad()
                tgt_seq, tgt_mask = seq['input_ids'].to(device), seq['attention_mask'].bool().to(device)
                encoded = encoder(im.to(device))
                loss = decoder(tgt_seq, mask=tgt_mask, context=encoded)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                dset.set_description('Loss: %.4f' % loss.item())
        #validation 
        evaluate(model, valdataloader, args, num_batches=int(args.valbatches*e/args.epochs), name='val')
        if (e+1) % args.save_freq == 0:
            save_models(e)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default='settings/config.yaml', help='path to yaml config file', type=argparse.FileType('r'))
    parser.add_argument('-d', '--data', default='dataset/data/train.pkl', type=str, help='Path to Dataset pkl file')
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--resume', help='path to checkpoint folder', action='store_true')

    parsed_args = parser.parse_args()
    with parsed_args.config as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)


    print('##############################################')
    print('training start')
    time1 = dt.datetime.now(gettz('Asia/Seoul'))
    print('current time:', time1.isoformat())
    train(args)
    time2 = dt.datetime.now(gettz('Asia/Seoul'))
    print('current time:', time2.isoformat())
    print('training end')
    print('##############################################')