'''
 run 'torchrun --standalone --nnodes=1 --nproc_per_node=2 run_train.py' to start training process with 2 GPUs
'''

import sys
sys.path.append('..')

from src.utils import set_random_seed
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import os
import random
from src.data.collator import Collator_fn, MoleculeDataset
from src.model.light import LiGhTPredictor as LiGhT
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.pretrain_trainer import Trainer
from src.trainer.result_tracker import Result_Tracker
import warnings
import pandas as pd
import json
from src.trainer.metrics import Evaluate
from src.model.contrast import Contrast

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12312'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
warnings.filterwarnings("ignore")

metric = Evaluate()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def cal_final_results(args):
    res = pd.read_csv(args.save_dir + '/results.csv')
    metrics = ['rmse', 'mae', 'sd', 'r']
    res_ls = {m: [] for m in metrics}
    try:
        for m in metrics:
            res_ls[m] = [float(res[res['repeat'] == i][m].values[0]) for i in range(args.n_repeat)]
    except:
        for m in metrics:
            res_ls[m] = [float(res[res['repeat'] == str(i)][m].values[0]) for i in range(args.n_repeat)]
    avg_res = [round(np.mean(res_ls[m]), 4) for m in metrics]
    std_res = [round(np.std(res_ls[m]), 4) for m in metrics]
    res.loc[len(res)] = ['avg'] + avg_res
    res.loc[len(res)] = ['std'] + std_res
    res.to_csv(args.save_dir + '/results.csv', index=False)
    print('Average results: ', avg_res)
    print(args.save_dir)


def load_snapshot(args, model, snapshot_path):
    snapshot = torch.load(snapshot_path, map_location=args.device)
    model.load_state_dict(snapshot["model_state"])
    args.epoch_run = snapshot["epoch_run"]
    try:
        args.step_count = snapshot['step_count']
    except:
        args.step_count = (len(train_loader.dataset) // args.batch_size) * args.epoch_run
    print(f"Resuming training from snapshot at Epoch {args.epoch_run}, step {args.step_count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--n_repeat", type=int, default=5)
    parser.add_argument("--dist_train", type=int, default=False, help='Distributed training')
    parser.add_argument("--only_test", type=int, default=False, help='Only test, no training process.')
    ## data path
    parser.add_argument("--dataset", type=str, default='PDBBind2020', choices=['PDBBind2020', 'CSAR'])
    parser.add_argument('--data_path', default='/media/data2/lm/Experiments/3D_DTI/dataset/', type=str)
    parser.add_argument('--refined_path', default='/original/PDBbind_v2020_refined/refined-set/', type=str, help='refined set path')
    parser.add_argument('--core_path', default='/original/CASF-2016/coreset/', type=str, help='core set path')
    parser.add_argument('--split_type', type=str, default='random_split', help='random split or temporal split')
    parser.add_argument('--processed_dir', type=str,
                        default='/media/data2/lm/Experiments/3D_DTI/GeoTrans_pool/dataset/processed/', help='Preprocessed dataset')
    ## dataset prepare
    parser.add_argument("--cutoff", type=int, default=5, help='threshold of atom distance')
    parser.add_argument("--inner_cutoff", type=int, default=5, help='threshold of atom distance')
    parser.add_argument("--n_angle", type=int, default=6, help='number of angle domains')
    parser.add_argument("--add_fea", type=int, default=0, help='add feature manner, 1, 2, others')
    parser.add_argument("--is_mask", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    ## model config
    # Basic config
    parser.add_argument("--graph_type", type=str, default='bab', help='b2b, bab, or both')
    parser.add_argument("--graph_pool", type=str, default='mean', choices=['sum', 'max', 'mean'], help='Pooling graph node')
    parser.add_argument("--pool", type=str, default="global_DiffPool",
                        choices=["global_SAGPool", "hier_SAGPool", "global_DiffPool", "hier_DiffPool", None],
                        help='Pooling operation')
    parser.add_argument("--pool_gnn", type=str, default='gat', choices=['graphsage', 'gcn', 'gat'], help='')
    parser.add_argument("--pool_rel", type=int, default=1, help='Using relatin features for pooling operation')
    parser.add_argument("--readout", type=str, default='last',
                        choices=['sum', 'max', 'mean', 'concat', 'last', 'linear', 'gru', 'lstm', 'bi-gru', 'bi-lstm'],
                        help='Readout operation for outputs of different layers')
    parser.add_argument("--pool_ratio", type=float, default=0.5, help='Pooling ratio for SAGPool')
    parser.add_argument("--pool_layer", type=int, default=1, help='Pooling layers for DiffPool')
    parser.add_argument("--assign_node", type=int, default=400, help='Assignment node for DiffPool')
    parser.add_argument("--init_emb", type=int, default=1, help='Embed node/edge embeddings')
    parser.add_argument("--n_layer", type=int, default=5, help='number of layers')
    parser.add_argument("--embed_type", type=str, default='float', help='int, float, both')
    parser.add_argument("--embed_dim", type=int, default=128, help='angle domain, bong length embedding dim')
    parser.add_argument("--hidden_size", type=int, default=128, help='head_dim * num_head')
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.)
    parser.add_argument("--input_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.)
    parser.add_argument("--feat_drop", type=float, default=0.)
    # Conifg for GNN
    parser.add_argument("--transformer", type=int, default=0, help='Use Transformer or GNN')
    parser.add_argument("--topk", type=int, default=-1, help='Top-K neighbors for each bond node')
    parser.add_argument("--diffusion", type=int, default=1, help='Attention diffusion or not')
    parser.add_argument("--layer_norm", type=int, default=1, help='Whether or not use LayerNorm in GNN')
    parser.add_argument("--leaky_relu", type=float, default=0.2, help='')
    parser.add_argument("--feed_forward", type=int, default=0, help='Whether or not use the FeedForward layer in GNN')
    parser.add_argument("--alpha_cal", type=int, default=1, help='Whether or not to calculate alpha in GNN')
    parser.add_argument("--n_hop", type=int, default=2, help='Number of attention diffusion hops in GNN')
    parser.add_argument("--residual", type=int, default=1, help='Whether or not to to use the residual connection in GNN')
    parser.add_argument("--is_rel", type=int, default=1, help='Whether the edge has representation')
    ## training config
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--n_epoch", type=int, default=300, help='Total epochs')
    parser.add_argument("--warmup_epoch", type=int, default=150, help='Warmup epochs')
    parser.add_argument("--epoch_run", type=int, default=0, help='Start epoch')
    parser.add_argument("--step_count", type=int, default=0, help='Start step')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5, help='weight decay')
    parser.add_argument("--patience", type=int, default=10, help='Early stopping')
    parser.add_argument("--max_norm", type=int, default=5, help='Clips gradient norm of an iterable of parameters.')
    ## save path
    parser.add_argument("--summary_writer", type=bool, default=0)
    parser.add_argument("--model_dir", type=str,
                        default=#'/media/data0/lm/Experiments/3D_DTI/GeoTrans/'
                                '../results/{}/trans{}_nlayer{}_cutoff{}_graph_pool{}_embed_type_{}_embed_dim{}_hsize{}'
                                '_nhead{}_{}_{}_readout_{}_lr{}_weight_decay{}_leaky{}_diff{}_{}_node{}_ln{}', )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    # # Distributed setting
    if args.dist_train:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        args.device = torch.device('cuda', local_rank)
    else:
        torch.cuda.set_device(args.device)
    set_random_seed(args.seed)

    # Save path
    args.model_dir = args.model_dir.format(args.split_type, args.transformer, args.n_layer, args.cutoff, args.graph_pool,
                                         args.embed_type, args.embed_dim, args.hidden_size, args.n_head, args.pool,
                                         args.pool_gnn, args.readout, args.lr, args.weight_decay, args.leaky_relu,
                                         args.diffusion, args.alpha_cal, args.assign_node, args.layer_norm)
    args.save_dir = args.model_dir# + '_CSAR/'
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + '/model/', exist_ok=True)
    os.makedirs(args.save_dir + '/tensorboard/', exist_ok=True)
    os.makedirs(args.save_dir + '/outputs/', exist_ok=True)


    sys.stdout = Logger(args.save_dir + 'log.txt')
    print(args.save_dir)
    f_csv = open(args.save_dir + '/results.csv', 'a')
    f_csv.write('repeat,rmse,mae,sd,r\n')
    f_csv.close()

    # Loss
    criterion = MSELoss(reduction='mean')

    for repeat in range(args.n_repeat):
        print(f'This is repeat {repeat}...')
        args.repeat = repeat
        args.epoch_run = 0
        # check wthether the model has been trained
        model_path = args.model_dir + f'/model/best_model_repeat{repeat}.pt'
        assert os.path.exists(model_path), 'The model does not exist!!!!'

        # Dataset prepare
        collator_test = Collator_fn(args, training=False)
        test_dataset = MoleculeDataset(graph_path=f'{args.processed_dir}/{args.dataset}/random_split/test_{args.cutoff}_{args.n_angle}_graph.pkl')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_threads,
                                 drop_last=False, worker_init_fn=seed_worker, collate_fn=collator_test)

        # Model
        model = LiGhT(args=args, d_hidden=args.hidden_size, n_layer=args.n_layer, n_heads=args.n_head, n_ffn_dense_layers=2,
                      input_drop=args.input_drop, attn_drop=args.attn_drop, feat_drop=args.feat_drop,
                      readout_mode=args.readout)
        model = model.to(args.device)
        load_snapshot(args, model, snapshot_path=model_path)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=7000, tot_updates=40000, step_count=args.step_count, lr=args.lr, end_lr=1e-9, power=1)

        if args.summary_writer:
            summary_writer = SummaryWriter(args.save_dir + f'/tensorboard/repest{repeat}')
        else:
            summary_writer = None

        trainer = Trainer(args, optimizer, lr_scheduler, criterion, summary_writer, device=args.device, local_rank=0)
        # the process may be terminated while training with distribution.
        # Then run 'killall python' in the terminal can kill the zombie processes
        preds, true_pks = trainer.predict(model, test_loader)
        rmse, mae, r, sd = metric.evaluate(true_pks, preds)
        print(round(rmse, 4), round(mae, 4), round(sd, 4), round(r, 4))
        res = dict()
        res['pred'] = preds.tolist()
        res['ground_truth'] = true_pks.tolist()
        json.dump(res, open(args.save_dir + f'/outputs/labels000{repeat}.json', 'w'))


        ls = [repeat, round(rmse, 4), round(mae, 4), round(sd, 4), round(r, 4)]
        f_csv = open(args.save_dir + '/results.csv', 'a')
        f_csv.write(','.join(map(str, ls)) + '\n')
        f_csv.close()
        if args.summary_writer:
            summary_writer.close()

    # compute mean and std
    cal_final_results(args)
    print('Done!!!!!!!!')

