from torch.autograd import Variable


def to_cuda_variable(tensor, use_cuda=True):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :param use_cuda: True, if cuda is available
    :return: torch Variable, of same size as tensor
    """
    if use_cuda:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


def to_cuda_variable_long(tensor, use_cuda=True):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :param use_cuda: True, if cuda is available
    :return: torch Variable, of same size as tensor
    """
    if use_cuda:
        return Variable(tensor.long()).cuda()
    else:
        return Variable(tensor.long())


def to_numpy(variable, use_cuda=True):
    """
    Converts torch Variable to numpy nd array
    :param variable: torch Variable, of any size
    :param use_cuda: True, if cuda is available
    :return: numpy nd array, of same size as variable
    """
    if use_cuda:
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()
