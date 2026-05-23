import os
import time
import random
import numpy as np
from numpy import random
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.optim as optim
from trainer.metrics import Metric
from models.bulid_model import build_model
from config.configurator import configs
from .utils import DisabledSummaryWriter, log_exceptions


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if 'log_loss' in configs['train'] and configs['train']['log_loss']:
            self.logger.log(loss_log_dict, save_to_log=False, print_to_console=True)

    # ---- opt-in resumable checkpointing -------------------------------------
    # Active ONLY when configs['train']['ckpt_out_dir'] is set (the resumable
    # single-run script sets it). When the key is absent the code path is
    # byte-identical to upstream, so the faithful 54-pt grid / single / d=128
    # re-validation sweeps are unaffected (no extra I/O, no resume, same RNG).
    def _ckpt_paths(self):
        d = configs['train'].get('ckpt_out_dir')
        if not d:
            return None, None
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, 'best_model.pt'), os.path.join(d, 'training_state.pt')

    def _save_training_state(self, model, ts_path, epoch_idx, best_epoch,
                             best_recall, now_patience, best_state_dict):
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        torch.save({
            'epoch': epoch_idx,
            'best_epoch': best_epoch,
            'best_recall': best_recall,
            'now_patience': now_patience,
            'model': model.state_dict(),
            'best_model': best_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': cuda_rng,
            'np_rng': np.random.get_state(),
            'hparams': {
                'embedding_size': configs['model'].get('embedding_size'),
                'batch_size': configs['train'].get('batch_size'),
                'seed': configs['train'].get('seed'),
            },
        }, ts_path)

    @log_exceptions
    def train(self, model):
        now_patience = 0
        best_epoch = 0
        best_recall = -1e9
        best_state_dict = None
        self.create_optimizer(model)
        train_config = configs['train']
        best_pt, ts_path = self._ckpt_paths()
        start_epoch = 0

        # resume from a prior training_state.pt if requested and present
        if ts_path and train_config.get('resume') and os.path.exists(ts_path):
            st = torch.load(ts_path, map_location=configs['device'],
                            weights_only=False)
            model.load_state_dict(st['model'])
            self.optimizer.load_state_dict(st['optimizer'])
            start_epoch = st['epoch'] + 1
            best_epoch = st['best_epoch']
            best_recall = st['best_recall']
            now_patience = st['now_patience']
            best_state_dict = st['best_model']
            try:
                torch.set_rng_state(st['torch_rng'].cpu()
                                    if hasattr(st['torch_rng'], 'cpu')
                                    else st['torch_rng'])
                if st.get('cuda_rng') is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(st['cuda_rng'])
                if st.get('np_rng') is not None:
                    np.random.set_state(st['np_rng'])
            except Exception as e:
                self.logger.log("RNG restore skipped ({}); continuing".format(e))
            self.logger.log(
                "Resumed from epoch {} (best_epoch={}, best_recall={:.4f}, "
                "patience={}/{})".format(start_epoch, best_epoch, best_recall,
                                         now_patience, train_config['patience']))

        for epoch_idx in range(start_epoch, train_config['epoch']):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result = self.evaluate(model, epoch_idx)

                if eval_result['recall'][-1] > best_recall:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_recall = eval_result['recall'][-1]
                    best_state_dict = deepcopy(model.state_dict())
                    if best_pt:
                        torch.save(best_state_dict, best_pt)
                        self.logger.log(
                            "Saved best_model.pt (epoch {}, recall@20={:.4f})"
                            .format(best_epoch, best_recall))
                else:
                    now_patience += 1

                # persist resumable state every evaluation
                if ts_path:
                    self._save_training_state(
                        model, ts_path, epoch_idx, best_epoch, best_recall,
                        now_patience, best_state_dict)

                # early stop
                if now_patience == configs['train']['patience']:
                    break

        # evaluation again
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        self.evaluate(model)

        # final test
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        test_result = self.test(model)

        # save result
        self.save_model(model)
        self.logger.log("Best Epoch {}. Final test result: {}.".format(best_epoch, test_result))

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        return eval_result
    
    @log_exceptions
    def test_save(self, model):
        model.eval()
        eval_result, candidate_set = self.metric.eval_save(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        return eval_result, candidate_set

    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            if not configs['tune']['enable']:
                save_dir_path = './encoder/checkpoint/{}'.format(model_name)

                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                torch.save(model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed']))
                self.logger.log("Save model parameters to {}".format('{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed'])))
            else:
                save_dir_path = './encoder/checkpoint/{}/tune'.format(model_name)

                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))

class AutoCFTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(AutoCFTrainer, self).__init__(data_handler, logger)
        self.fix_steps = configs['model']['fix_steps']

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            if i % self.fix_steps == 0:
                sampScores, seeds = model.sample_subgraphs()
                encoderAdj, decoderAdj = model.mask_subgraphs(seeds)

            loss, loss_dict = model.cal_loss(batch_data, encoderAdj, decoderAdj)

            if i % self.fix_steps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
                loss_dict['infomax_loss'] = localGlobalLoss

            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        # writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)



