import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Optional path to a YAML config file overriding the default '
                             './encoder/config/modelconf/{model}.yml. Used by the '
                             'ML-20M capacity-match wrappers (run_ml20m_r{,plus}.sh) to '
                             'inject embedding_size=128 without modifying the vendored '
                             'upstream YAML (which keeps embedding_size=32 for the '
                             'sparse-datapoint runs). When unset, behaviour is identical '
                             'to upstream RLMRec.')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file (default = upstream-vendored location; --model-conf overrides)
    yml_path = args.model_conf or './encoder/config/modelconf/{}.yml'.format(model_name)
    if not os.path.exists(yml_path):
        raise Exception(f"Config YAML not found: {yml_path}")

    # read yml file
    with open(yml_path, encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed

        # semantic embeddings
        usrprf_embeds_path = "./data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        itmprf_embeds_path = "./data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        with open(usrprf_embeds_path, 'rb') as f:
            configs['usrprf_embeds'] = pickle.load(f)
        with open(itmprf_embeds_path, 'rb') as f:
            configs['itmprf_embeds'] = pickle.load(f)

        return configs

configs = parse_configure()
