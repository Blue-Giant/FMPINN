import torch


def save_torch_net2file(outPath=None, model2net=None, name2model='DNN', opt_way2save='para'):
    """
    Args:
         outPath: the output path for saving net model
         model2net: the net model required to save
         opt_way2save: the optional way to save model
    """
    if str.lower(opt_way2save) == 'para':
        outFile_with_path = '%s/%s.pth' % (outPath, str.lower(name2model))
        torch.save(model2net.state_dict(), outFile_with_path)
    else:
        outFile_with_path = '%s/%s.pth' % (outPath, str.lower(name2model))
        torch.save(model2net(), outFile_with_path)


def load_torch_net2file(path2file=None, model2net=None, name2model='DNN', opt_way2load='para'):
    """
    Args:
         path2file: the path for loading net model
         model2net: the net model required to load
         opt_way2load: the option to load model
    """
    file2model_with_path = '%s/%s.pth' % (path2file, str.lower(name2model))
    if str.lower(opt_way2load) == 'para':
        # 这种方式加载模型参数，需要提前定义好一个和参数匹配的模型
        model2net.load_state_dict(torch.load(file2model_with_path))
    else:
        model2net = torch.load(file2model_with_path)

    return model2net


def save_torch_net2file_with_keys(outPath=None, model2net=None, name2model='DNN', optimizer=None, scheduler=None,
                                  epoch=100, loss=0.1):
    """
    Args:
         outPath: the output path for saving net model
         model2net: the net model required to save
         name2model: the name of network
         optimizer: the optimizer, such as Adam, Moment, SGD....
         scheduler: a scheduler used to adjust learning rate for optimizer
         epoch: current training epoch
         loss: current training loss
    """
    outFile_with_path = '%s/%s.pth' % (outPath, str.lower(name2model))
    torch.save(
        {'epoch': epoch,
         'model': model2net.state_dict(),
         'optimizer': optimizer.state_dict(),
         'scheduler': scheduler.state_dict(),
         'loss': loss}, outFile_with_path)


def load_torch_net2file_with_keys(path2file=None, model2net=None, name2model='DNN', optimizer=None, scheduler=None):
    """
    Args:
         path2file: the path for loading net model
         model2net: the net model required to load
         name2model: the name of network
         optimizer: the optimizer, such as Adam, Moment, SGD....
         scheduler: a scheduler used to adjust learning rate for optimizer
    """
    file2model_with_path = '%s/%s.pth' % (path2file, str.lower(name2model))
    checkpoint = torch.load(file2model_with_path)
    model2net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def load_torch_net_and_epoch(path2file=None, model2net=None, name2model='DNN'):
    """
    Args:
         path2file: the path for loading net model
         model2net: the net model required to load
         name2model: the name of network
    """
    file2model_with_path = '%s/%s.pth' % (path2file, str.lower(name2model))
    checkpoint = torch.load(file2model_with_path)
    model2net.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    return epoch



