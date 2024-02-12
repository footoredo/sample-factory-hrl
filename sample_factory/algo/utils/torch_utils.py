from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config


def init_torch_runtime(cfg: AttrDict, max_num_threads: Optional[int] = 1):
    torch.multiprocessing.set_sharing_strategy("file_system")
    if max_num_threads is not None:
        torch.set_num_threads(max_num_threads)
    if cfg.device == "gpu":
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = True


def inference_context(is_serial):
    if is_serial:
        # in serial mode we use the same tensors on sampler and learner
        return torch.no_grad()
    else:
        return torch.inference_mode()


def to_torch_dtype(numpy_dtype):
    """from_numpy automatically infers type, so we leverage that."""
    x = np.zeros([1], dtype=numpy_dtype)
    t = torch.from_numpy(x)
    return t.dtype


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


def to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value


@torch.jit.script
def masked_select(x: torch.Tensor, mask: torch.Tensor, num_non_mask: int) -> torch.Tensor:
    if num_non_mask == 0:
        return x
    else:
        return torch.masked_select(x, mask)


def synchronize(cfg: Config, device: torch.device | str) -> None:
    if cfg.serial_mode:
        return

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        
        
def recycle_weights(module: torch.nn.Linear, neuron_mask):
    if neuron_mask.sum() > 0:
        weight_shape = module.weight.shape
        new_weights = torch.empty(weight_shape, device=module.weight)
        torch.nn.init.uniform_(new_weights, -np.sqrt(1 / weight_shape[0]), np.sqrt(1 / weight_shape[0]))


class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        

# from DrM (https://github.com/XuGW-Kevin/DrM/blob/main/utils.py#L156)

def cal_dormant_ratio(model, *inputs, percentage=0.025, recycle=False):
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0

    for module in model.modules():
        # print(module)
        if isinstance(module, torch.nn.Linear):
            # (isinstance(module, torch.jit.RecursiveScriptModule) and module.original_name == "Linear"):  # jit
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        outputs = model(*inputs)

    for module, hook in zip(
        (module
         for module in model.modules() if isinstance(module, torch.nn.Linear)),
            hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indices = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indices)
                print(output_data.shape, avg_neuron_output, module.weight.shape)

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    return dormant_neurons, total_neurons, outputs
