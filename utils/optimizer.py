import pdb
import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model,
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            alpha = self.optim_dict['learning_ratio']
            if self.optim_dict['bert_different_layer_lr'] == True:
                all_params = model.parameters()
                params_3 = []
                params_2 = []
                params_1 = []
                # 根据自己的筛选规则  将所有网络参数进行分组
                for pname, p in model.named_parameters():
                    if any([k in pname for k in ['multimodal_fusion.bert.encoder.layer.0', 'multimodal_fusion.bert.encoder.layer.1', 'multimodal_fusion.bert.encoder.layer.2', 'multimodal_fusion.bert.encoder.layer.3']]):
                        params_2 += [p]
                        print('------------')
                    elif any([k in pname for k in ['multimodal_fusion.bert.encoder.layer.4', 'multimodal_fusion.bert.encoder.layer.5']]):
                        params_1 += [p]
                        print('------------')
                    elif any([k in pname for k in ['bert_text.bert.encoder.layer']]):
                        params_3 += [p]
                        print('------------')
                    print(pname)
                    # print(p)  
                # 取回分组参数的id
                params_id = list(map(id, params_3)) + list(map(id, params_2)) + list(map(id, params_1))
                # 取回剩余分特殊处置参数的id
                other_params = list(filter(lambda p: id(p) not in params_id, all_params))

                self.optimizer = optim.Adam(
                    [   
                        {'params': other_params, 'lr': self.optim_dict['base_lr']},
                        {'params': params_1, 'lr': self.optim_dict['base_lr']* 0.1},
                        {'params': params_2, 'lr': self.optim_dict['base_lr']* 0.01},
                        {'params': params_3, 'lr': self.optim_dict['base_lr']* 0.001},
                        
                    ],
                    # model.conv1d.fc.parameters(),
                    # model.parameters(),
                    # lr=self.optim_dict['base_lr'],
                    weight_decay=self.optim_dict['weight_decay']
                )
            elif self.optim_dict['bert_different_layer_lr'] == 'freezeSign':
                all_params = model.parameters()
                params_sign = []
                params_text = []
                # 根据自己的筛选规则  将所有网络参数进行分组
                for pname, p in model.named_parameters():
                    if any([k in pname for k in ['sign_encoder']]):
                        params_sign += [p]
                        p.requires_grad = False
                        print('------------')
                    elif any([k in pname for k in ['bert_text.bert.encoder.layer']]):
                        params_text += [p]
                        print('------------')
                    print(pname)
                    # print(p)  
                # 取回分组参数的id
                params_id = list(map(id, params_sign)) + list(map(id, params_text))
                # 取回剩余分特殊处置参数的id
                other_params = list(filter(lambda p: id(p) not in params_id, all_params))

                self.optimizer = optim.Adam(
                    [   
                        {'params': other_params, 'lr': self.optim_dict['base_lr']},
                        {'params': params_sign, 'lr': self.optim_dict['base_lr']* 0.001},
                        {'params': params_text, 'lr': self.optim_dict['base_lr']* 0.001},
                        
                    ],
                    # model.conv1d.fc.parameters(),
                    # model.parameters(),
                    # lr=self.optim_dict['base_lr'],
                    weight_decay=self.optim_dict['weight_decay']
                )
            elif self.optim_dict['bert_different_layer_lr'] == 'freezeText':
                all_params = model.parameters()
                params_sign = []
                params_text = []
                # 根据自己的筛选规则  将所有网络参数进行分组
                for pname, p in model.named_parameters():
                    if any([k in pname for k in ['sign_encoder']]):
                        params_sign += [p]
                        # p.requires_grad = False
                        print('------------')
                    elif any([k in pname for k in ['bert_text.bert.encoder.layer']]):
                        params_text += [p]
                        p.requires_grad = False
                        print('------------')
                    print(pname)
                    # print(p)  
                # 取回分组参数的id
                params_id = list(map(id, params_sign)) + list(map(id, params_text))
                # 取回剩余分特殊处置参数的id
                other_params = list(filter(lambda p: id(p) not in params_id, all_params))

                self.optimizer = optim.Adam(
                    [   
                        {'params': other_params, 'lr': self.optim_dict['base_lr']},
                        {'params': params_sign, 'lr': self.optim_dict['base_lr']},
                        {'params': params_text, 'lr': self.optim_dict['base_lr']},
                        
                    ],
                    # model.conv1d.fc.parameters(),
                    # model.parameters(),
                    # lr=self.optim_dict['base_lr'],
                    weight_decay=self.optim_dict['weight_decay']
                )
            elif self.optim_dict['bert_different_layer_lr'] == 'freezeTextAndSign':
                all_params = model.parameters()
                params_sign = []
                params_text = []
                # 根据自己的筛选规则  将所有网络参数进行分组
                for pname, p in model.named_parameters():
                    if any([k in pname for k in ['sign_encoder']]):
                        params_sign += [p]
                        p.requires_grad = False
                        print('------------')
                    elif any([k in pname for k in ['bert_text.bert.encoder.layer']]):
                        params_text += [p]
                        p.requires_grad = False
                        print('------------')
                    print(pname)
                    # print(p)  
                # 取回分组参数的id
                params_id = list(map(id, params_sign)) + list(map(id, params_text))
                # 取回剩余分特殊处置参数的id
                other_params = list(filter(lambda p: id(p) not in params_id, all_params))

                self.optimizer = optim.Adam(
                    [   
                        {'params': other_params, 'lr': self.optim_dict['base_lr']},
                        {'params': params_sign, 'lr': self.optim_dict['base_lr']},
                        {'params': params_text, 'lr': self.optim_dict['base_lr']},
                        
                    ],
                    # model.conv1d.fc.parameters(),
                    # model.parameters(),
                    # lr=self.optim_dict['base_lr'],
                    weight_decay=self.optim_dict['weight_decay']
                )
            else:
                self.optimizer = optim.Adam(
                    # [
                    #     {'params': model.conv2d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                    #     {'params': model.conv1d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                    #     {'params': model.rnn.parameters()},
                    #     {'params': model.classifier.parameters()},
                    # ],
                    # model.conv1d.fc.parameters(),
                    model.parameters(),
                    lr=self.optim_dict['base_lr'],
                    weight_decay=self.optim_dict['weight_decay']
                )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
