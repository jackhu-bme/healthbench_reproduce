import argparse
import logging
import shlex
import subprocess
import time
import json
from dataclasses import dataclass
from io import StringIO

import torch
from torchvision.models import resnet101

_logger = logging.getLogger(__name__)


def occupy_gpu():
    batch_size = 40
    _logger.info('start running resnet101')
    model = resnet101().cuda()
    model = torch.nn.DataParallel(model)
    device = torch.device('cuda')
    num_gpus = torch.cuda.device_count()

    while True:
        x = torch.rand(batch_size * num_gpus, 3, 512, 512, device=device)
        y = model(x)
        time.sleep(0.05)

def main():
    occupy_gpu()

if __name__ == "__main__":
    main()
