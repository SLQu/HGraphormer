
def setEnv(args):
    import torch
    import numpy as np
    import os, sys

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    return torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')


def gpuInfo(baselogger):
    import pynvml
    pynvml.nvmlInit()

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = meminfo.total / (1024 ** 3)
    used = meminfo.used / (1024 ** 3)
    rest = meminfo.free / (1024 ** 3)

    baselogger.info(f'Device {pynvml.nvmlDeviceGetName(handle)}')
    deviceCount = pynvml.nvmlDeviceGetCount()  # number of gpu
    baselogger.info(f'gpu number {deviceCount}')
    baselogger.info(f'total gpu memory: {total} G.')
    baselogger.info(f'used gpu memory:  {used} G.')
    baselogger.info(f'rest gpu memory:  {rest} G.')


def storeFile(res_dir):
    import shutil
    shutil.copy('utils.py', res_dir)
    shutil.copy('readme.md', res_dir)
    shutil.copy('config.py', res_dir)
    shutil.copy('logger.py', res_dir)
    shutil.copy('main.py', res_dir)
    # shutil.copy('datasets.py', res_dir)

    # shutil.copy('static.py', res_dir)
    # shutil.copy('prepare.py', res_dir)
    # shutil.copy('sampling.py', res_dir)
    # shutil.copy('transformer.py', res_dir)


def ckeckMakeDir(_path_):
    import os.path as osp
    import os

    def check_dir(folder, mk_dir=True):
        if not osp.exists(folder):
            if mk_dir:
                print(f'making direction {folder}!')
                os.mkdir(folder)
            else:
                raise Exception(f'Not exist direction {folder}')


    folders = _path_.split(os.sep)[1:]
    root = ''
    for f in folders:
        root += os.sep +f
        check_dir(root)