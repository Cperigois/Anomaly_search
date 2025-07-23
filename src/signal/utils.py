import numpy as np
import torch

def time_conversion(value, input_unit, target_unit):
    """
    Time conversion from s, min, h, d, w, month, y to s, min, h, d, w, month, y.
    Start with a conversion of the input in second and then convert the seconds in the wanted unit.
    :param value: (float) input value.
    :param input_unit: (str) input_value unit should be among {'s','min','h','d','w','month','y'}
    :param target_unit: (str) output value unit should be among {'s','min','h','d','w','month','y'}
    :return: (float) the value in the target unit.
    """
    # First convert input value to seconds
    if input_unit == 's':
        seconds = value
    elif input_unit == 'min':
        seconds = value * 60
    elif input_unit == 'h':
        seconds = value * 3600
    elif input_unit == 'd':
        seconds = value * 86400
    elif input_unit == 'w':
        seconds = value * 604800
    elif input_unit == 'month':
        # Assuming 1 month = 30 days
        seconds = value * 2592000
    elif input_unit == 'y':
        # Assuming 1 year = 365 days (not accounting for leap years)
        seconds = value * 31536000
    else:
        raise ValueError(f"Invalid input unit: {input_unit}. Must be one of {'s','min','h','d','w','month','y'}")

    # Then convert seconds to target unit
    if target_unit == 's':
        return seconds
    elif target_unit == 'min':
        return seconds / 60
    elif target_unit == 'h':
        return seconds / 3600
    elif target_unit == 'd':
        return seconds / 86400
    elif target_unit == 'w':
        return seconds / 604800
    elif target_unit == 'month':
        return seconds / 2592000
    elif target_unit == 'y':
        return seconds / 31536000
    else:
        raise ValueError(f"Invalid target unit: {target_unit}. Must be one of {'s','min','h','d','w','month','y'}")

def mse_loss(output, target):
    if output.shape[-1] > target.shape[-1]:
        output = output[..., :target.shape[-1]]
    elif output.shape[-1] < target.shape[-1]:
        target = target[..., :output.shape[-1]]
    return torch.mean((output - target) ** 2)
