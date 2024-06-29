import os
import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        device = str(device)
        if device != 'None':
            self.gpu_list = [i for i in range(len(device.split(',')))]
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            output_device = self.gpu_list[0]
            self.occupy_gpu(self.gpu_list)
        self.output_device = output_device if len(self.gpu_list) > 0 else "cpu"

    def model_to_device(self, model):
        # model = convert_model(model)
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        return data.to(self.output_device)  #


    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi.
        """
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)
