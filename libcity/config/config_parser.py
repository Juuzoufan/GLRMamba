import os
import json
import torch


class ConfigParser(object):

    def __init__(self, task, model, dataset, config_file=None,
                 saved_model=True, train=None, other_args=None, hyper_config_dict=None, initial_ckpt=None):
        self.config = {}
        # 先读取配置文件，获取可能的 task/model/dataset/train 值
        self._parse_config_file(config_file)
        # 再解析外部参数（命令行参数优先，但 None 值会使用配置文件的值）
        self._parse_external_config(task, model, dataset, saved_model, train, other_args, hyper_config_dict, initial_ckpt)
        self._load_default_config()
        self._init_device()

    def _parse_external_config(self, task, model, dataset,
                               saved_model=True, train=None, other_args=None, hyper_config_dict=None, initial_ckpt=None):
        # 如果命令行没有指定（为 None），则使用配置文件中的值
        if task is not None:
            self.config['task'] = task
        if model is not None:
            self.config['model'] = model
        if dataset is not None:
            self.config['dataset'] = dataset
        
        # 验证必需参数
        if self.config.get('task') is None:
            raise ValueError('the parameter task should not be None!')
        if self.config.get('model') is None:
            raise ValueError('the parameter model should not be None!')
        if self.config.get('dataset') is None:
            raise ValueError('the parameter dataset should not be None!')
        
        self.config['saved_model'] = saved_model
        # train 参数：命令行 > 配置文件 > 默认值 True
        if train is not None:
            self.config['train'] = train
        elif 'train' not in self.config:
            self.config['train'] = True
        # 特殊处理 map_matching 任务
        if self.config['task'] == 'map_matching':
            self.config['train'] = False
            
        if other_args is not None:
            for key in other_args:
                self.config[key] = other_args[key]
        if hyper_config_dict is not None:
            for key in hyper_config_dict:
                self.config[key] = hyper_config_dict[key]
        if initial_ckpt is not None:
            self.config['initial_ckpt'] = initial_ckpt
        elif 'initial_ckpt' not in self.config:
            self.config['initial_ckpt'] = None

    def _parse_config_file(self, config_file):
        if config_file is not None:
            # 支持绝对路径和相对路径
            if os.path.isabs(config_file):
                # 绝对路径：直接使用（如果没有.json后缀则添加）
                if not config_file.endswith('.json'):
                    config_file = config_file + '.json'
                config_path = config_file
            else:
                # 相对路径：在当前目录下查找
                config_path = './{}.json'.format(config_file)
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    x = json.load(f)
                    for key in x:
                        # 配置文件的所有值都先加载到 config 中
                        self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {} is not found. Please ensure \
                    the config file exists and is a JSON \
                    file.'.format(config_path))

    def _load_default_config(self):
        with open('./libcity/config/task_config.json', 'r') as f:
            task_config = json.load(f)
            if self.config['task'] not in task_config:
                raise ValueError(
                    'task {} is not supported.'.format(self.config['task']))
            task_config = task_config[self.config['task']]
            if self.config['model'] not in task_config['allowed_model']:
                raise ValueError('task {} do not support model {}'.format(
                    self.config['task'], self.config['model']))
            model = self.config['model']
            if 'dataset_class' not in self.config:
                self.config['dataset_class'] = task_config[model]['dataset_class']
            if self.config['task'] == 'traj_loc_pred' and 'traj_encoder' not in self.config:
                self.config['traj_encoder'] = task_config[model]['traj_encoder']
            if 'executor' not in self.config:
                self.config['executor'] = task_config[model]['executor']
            if 'evaluator' not in self.config:
                self.config['evaluator'] = task_config[model]['evaluator']
            if self.config['model'].upper() in ['LSTM', 'GRU', 'RNN']:
                self.config['rnn_type'] = self.config['model']
                self.config['model'] = 'RNN'
        default_file_list = []
        default_file_list.append('model/{}/{}.json'.format(self.config['task'], self.config['model']))
        default_file_list.append('data/{}.json'.format(self.config['dataset_class']))
        default_file_list.append('executor/{}.json'.format(self.config['executor']))
        default_file_list.append('evaluator/{}.json'.format(self.config['evaluator']))
        for file_name in default_file_list:
            with open('./libcity/config/{}'.format(file_name), 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]
        with open('./raw_data/{}/config.json'.format(self.config['dataset']), 'r') as f:
            x = json.load(f)
            for key in x:
                if key == 'info':
                    for ik in x[key]:
                        if ik not in self.config:
                            self.config[ik] = x[key][ik]
                else:
                    if key not in self.config:
                        self.config[key] = x[key]

    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        distributed = False
        if 'WORLD_SIZE' in os.environ:
            distributed = int(os.environ['WORLD_SIZE']) > 1
        self.config['distributed'] = distributed
        if use_gpu and distributed:
            local_rank = self.config['local_rank']
            assert local_rank >= 0
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            rank = torch.distributed.get_rank()
            self.config["rank"] = rank
            assert rank >= 0
            self.config['world_size'] = torch.distributed.get_world_size()
            self.config['device'] = torch.device(
                "cuda:%d" % local_rank if torch.cuda.is_available() else "cpu")
        else:
            if use_gpu:
                torch.cuda.set_device(0)
            self.config['device'] = torch.device(
                "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return self.config.__iter__()
