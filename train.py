import argparse
import os
from itertools import chain

from icecream import install
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from ogb.graphproppred.mol_encoder import AtomEncoder

from commons.utils import seed_all, get_random_indices, TENSORBOARD_FUNCTIONS
from datasets.chembl_dataset import ChEMBLDataset
from datasets.ogbg_dataset_extension import OGBGDatasetExtension

import seaborn

install()
seaborn.set_theme()

import torch
import yaml
from models import *
from datasets.custom_collate import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove

from torch.utils.data import DataLoader, Subset

from trainer.metrics import Rsquared, NegativeSimilarity, MeanPredictorLoss, \
    PositiveSimilarity, ContrastiveAccuracy, TrueNegativeRate, TruePositiveRate, Alignment, Uniformity, \
    BatchVariance, DimensionCovariance, MAE, PositiveSimilarityMultiplePositivesSeparate2d, \
    NegativeSimilarityMultiplePositivesSeparate2d, OGBEvaluator, PearsonR, PositiveProb, NegativeProb
from trainer.trainer import Trainer


def get_trainer(args, model, data, device, metrics):
    tensorboard_functions = {function: TENSORBOARD_FUNCTIONS[function] for function in args.tensorboard_functions}
    if args.dataset =='chembl':
        return PreTrainer(model=model, args=args, metrics=metrics, main_metric=args.main_metric,
                       main_metric_goal=args.main_metric_goal, optim=globals()[args.optimizer],
                       loss_func=globals()[args.loss_func](**args.loss_params), device=device,
                       tensorboard_functions=tensorboard_functions,
                       scheduler_step_per_batch=args.scheduler_step_per_batch)
    return Trainer(model=model, args=args, metrics=metrics, main_metric=args.main_metric,
                   main_metric_goal=args.main_metric_goal, optim=globals()[args.optimizer],
                   loss_func=globals()[args.loss_func](**args.loss_params), device=device,
                   tensorboard_functions=tensorboard_functions, scheduler_step_per_batch=args.scheduler_step_per_batch)


def load_model(args, data, device):
    model = globals()[args.model_type](device=device, **args.model_parameters)
    if args.pretrain_checkpoint:
        # get arguments used during pretraining
        with open(os.path.join(os.path.dirname(args.pretrain_checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            pretrain_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        pretrain_args = argparse.Namespace()
        pretrain_args.__dict__.update(pretrain_dict)

        checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
        # get all the weights that have something from 'args.transfer_layers' in their keys name
        # but only if they do not contain 'teacher' and remove 'student.' which we need for loading from BYOLWrapper
        pretrained_gnn_dict = {k.replace('student.', ''): v for k, v in checkpoint['model_state_dict'].items() if any(
            transfer_layer in k for transfer_layer in args.transfer_layers) and 'teacher' not in k and not any(
            to_exclude in k for to_exclude in args.exclude_from_transfer)}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_gnn_dict)  # update the gnn layers with the pretrained weights
        model.load_state_dict(model_state_dict)
        return model, pretrain_args.num_train
    return model, None


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    metrics_dict = {'rsquared': Rsquared(),
                    'mae': MAE(),
                    'pearsonr': PearsonR(),
                    'ogbg-molhiv': OGBEvaluator(d_name='ogbg-molhiv', metric='rocauc'),
                    'ogbg-molpcba': OGBEvaluator(d_name='ogbg-molpcba', metric='ap'),
                    'ogbg-molbace': OGBEvaluator(d_name='ogbg-molbace', metric='rocauc'),
                    'ogbg-molbbbp': OGBEvaluator(d_name='ogbg-molbbbp', metric='rocauc'),
                    'ogbg-molclintox': OGBEvaluator(d_name='ogbg-molclintox', metric='rocauc'),
                    'ogbg-moltoxcast': OGBEvaluator(d_name='ogbg-moltoxcast', metric='rocauc'),
                    'ogbg-moltox21': OGBEvaluator(d_name='ogbg-moltox21', metric='rocauc'),
                    'ogbg-mollipo': OGBEvaluator(d_name='ogbg-mollipo', metric='rmse'),
                    'ogbg-molmuv': OGBEvaluator(d_name='ogbg-molmuv', metric='ap'),
                    'ogbg-molsider': OGBEvaluator(d_name='ogbg-molsider', metric='rocauc'),
                    'ogbg-molfreesolv': OGBEvaluator(d_name='ogbg-molfreesolv', metric='rmse'),
                    'ogbg-molesol': OGBEvaluator(d_name='ogbg-molesol', metric='rmse'),
                    'positive_similarity': PositiveSimilarity(),
                    'positive_similarity_multiple_positives_separate2d': PositiveSimilarityMultiplePositivesSeparate2d(),
                    'positive_prob': PositiveProb(),
                    'negative_prob': NegativeProb(),
                    'negative_similarity': NegativeSimilarity(),
                    'negative_similarity_multiple_positives_separate2d': NegativeSimilarityMultiplePositivesSeparate2d(),
                    'contrastive_accuracy': ContrastiveAccuracy(threshold=0.5009),
                    'true_negative_rate': TrueNegativeRate(threshold=0.5009),
                    'true_positive_rate': TruePositiveRate(threshold=0.5009),
                    'mean_predictor_loss': MeanPredictorLoss(globals()[args.loss_func](**args.loss_params)),
                    'uniformity': Uniformity(t=2),
                    'alignment': Alignment(alpha=2),
                    'batch_variance': BatchVariance(),
                    'dimension_covariance': DimensionCovariance()
                    }
    print('using device: ', device)

    train_ogbg(args, device, metrics_dict)


def train_ogbg(args, device, metrics_dict):
    if args.dataset == 'chembl':
        dataclass = ChEMBLDataset
    else:
        dataclass = OGBGDatasetExtension
    dataset = dataclass(return_types=args.required_data, device=device, name=args.dataset)

    if args.dataset == 'chembl':
        all_idx = get_random_indices(len(dataset), args.seed_data)
        model_idx = all_idx[:380000]
        test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(dataset))]
        val_idx = all_idx[len(model_idx) + len(test_idx):]
        train_idx = model_idx[:args.num_train]

        split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
    else:
        split_idx = dataset.get_idx_split()

    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)

    train_loader = DataLoader(Subset(dataset, split_idx["train"]), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_function)
    val_loader = DataLoader(Subset(dataset, split_idx["valid"]), batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_function)
    test_loader = DataLoader(Subset(dataset, split_idx["test"]), batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_function)

    model, num_pretrain = load_model(args, data=dataset, device=device)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    if args.dataset != 'chembl':
        metrics[args.dataset] = metrics_dict[args.dataset]
        args.main_metric = args.dataset
        args.val_per_batch = False
        args.main_metric_goal = 'min' if metrics[args.main_metric].metric == 'rmse' else 'max'
    trainer = get_trainer(args=args, model=model, data=dataset, device=device, metrics=metrics)
    trainer.train(train_loader, val_loader)

    if args.eval_on_test:
        trainer.evaluation(test_loader, data_split='test')


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/pretrain.yml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--prefetch_graphs', type=bool, default=True,
                   help='load graphs into memory (needs RAM and upfront computation) for faster data loading during training')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset', type=str, default='qm9', help='[qm9, zinc, drugs, geom_qm9, molhiv]')
    p.add_argument('--num_train', type=int, default=100000, help='n samples of the model samples to use for train')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--critic_loss', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--critic_loss_params', type=dict, default={},
                   help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--linear_probing_samples', type=int, default=500,
                   help='number of samples to use for linear probing in the run_eval_per_epoch function of the self supervised trainer')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='mae_denormalized', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--pretrain_checkpoint', type=str, help='Specify path to finetune from a pretrained checkpoint')
    p.add_argument('--transfer_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--exclude_from_transfer', default=[],
                   help='parameters that usually should not be transferred like batchnorm params')
    p.add_argument('--transferred_lr', type=float, default=None, help='set to use a different LR for transfer layers')

    p.add_argument('--pos_dir', type=bool, default=False, help='adds pos dir as key to dgl graphs (required for dgn)')
    p.add_argument('--required_data', default=[],
                   help='what will be included in a batch like [mol_graph, targets, mol_graph3d]')
    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--use_e_features', default=True, type=bool, help='ignore edge features if set to False')
    p.add_argument('--targets', default=[], help='properties that should be predicted')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--dist_embedding', type=bool, default=False, help='add dist embedding to complete graphs edges')
    p.add_argument('--num_radial', type=int, default=6, help='number of frequencies for distance embedding')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--model3d_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--model3d_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--critic_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--critic_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--ssl_mode', type=str, default='contrastive', help='[contrastive, byol, alternating, philosophy]')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--transfer_from_different_dataset', type=bool, default=False,
                   help='set to true when transferring from different dataset')

    args = p.parse_args()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}

    if args.checkpoint:  # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
    return args


if __name__ == '__main__':
    train(parse_arguments())
