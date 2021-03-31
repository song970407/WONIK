def rescale_data(data, scaler):
    """
    It only works for minmax scaler.
    :param data: torch.Tensor or numpy.ndarray
    :param scaler: tuple of float, scaler[0] = minimum value, scaler[1] = maximum value
    :return:
    """
    return data * (scaler[1] - scaler[0]) + scaler[0]