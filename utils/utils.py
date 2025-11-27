import torch
import math
import gc
from typing import Optional

def cleanup() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * math.ceil(val1 / val2)) == val1

# Map a Pytorch dtype into a safetensor dtype
def encode_safetensor_type(data):
    if isinstance(data, (torch.Tensor, torch.nn.Parameter)):
        return data
    if isinstance(data, torch.Size):
        return torch.tensor(data)
    if isinstance(data, torch.dtype):
        data = str(data)
    if isinstance(data, bool):
        return torch.tensor(int(data), dtype=torch.uint8)
    if isinstance(data, int):
        return torch.tensor(data, dtype=torch.int32)
    if isinstance(data, float):
        return torch.tensor(data, dtype=torch.float32)
    if isinstance(data, str):
        return torch.tensor([ord(i) for i in data], dtype=torch.uint8)
    
# Decode a safetensor dtype into a Pytorch dtype
def decode_safetensor_type(data, data_type):
    if data_type in [torch.Tensor, torch.nn.Parameter]:
        return data
    if data_type is torch.Size:
        return torch.Size(data)
    if data_type is bool:
        return bool(data.item())
    if data_type is int:
        return int(data.item())
    if data_type is float:
        return float(data.item())
    if data_type is str:
        return "".join([chr(i) for i in data])
    if data_type is torch.dtype:
        return eval("".join([chr(i) for i in data]))

def replace_linear(
    model: torch.nn.Module,
    linear_replacement: torch.nn.Module,
    skip_modules=("lm_head"),
    copy_weights: bool = False,
    post_processing_function: Optional[str] = None,
) -> torch.nn.Module:
    """
    遍历一个模型，把其中的 torch.nn.Linear 层替换为新的线性层。

    参数：
        model (`torch.nn.Module`):
            要处理的模型（或者子模块）；函数会递归遍历所有子模块。
        linear_replacement (`torch.nn.Module` or callable):
            用来替换原来 Linear 的新 Linear 类／构造函数。
            如果需要传额外参数，可传入 lambda 或者偏函数等方式封装。
        skip_modules (`tuple[str]`, 可选，默认为 ("lm_head",)):
            模块名字里如果包含这些字符串，就不替换它们。
            通常像 `lm_head`（语言模型的首输出线性层）这样的层是敏感的，
            替换可能导致输出维度／损失计算不一致／行为变差，所以默认跳过。
        copy_weights (`bool`):
            如果为 True，替换后的新线性层会把原来的权重 (weight) 和偏置 (bias) 从旧模块复制过来。
            如果为 False，就用新层的初始化权重／bias。
        post_processing_function (`str` 可选):
            替换后如果新模块需要做一些额外处理／初始化，可以传入这个名字。
            函数会尝试在旧模块上调用 `getattr(module, post_processing_function)`，
            如果存在就执行这个函数。这个参数用于比如某些 Linear 层创建时需要额外设置的情形。

    返回：
        返回替换后的模型（原地修改），所有不在 skip_modules 中的 Linear 层均被替换。

    """
    # 遍历 model 的直接子模块（named_children 提供“名字 + 子模块 对”）
    for name, module in model.named_children():
        # 如果子模块本身还有子模块（module.children 非空），递归调用替换函数
        # 这样可以深入所有层级（Transformer block、子 layer 等）
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules, copy_weights, post_processing_function)
        # 判断这个 module 是否是标准的 Linear 且名字不在 skip_modules 中
        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            # 保存旧的 Linear 模块（用于权重复制／bias 复制等）
            old_module = model._modules[name]

            # 用新的 Linear 替换它；linear_replacement 通常是一个 class 或构造函数／lambda
            # 这里按标准 Linear 的构造参数 in_features, out_features, bias 是否存在
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
        
            # 如果 copy_weights 为 True，把旧模块的 weight 和 bias 复制到新模块里
            if copy_weights:
                # 要注意：这里直接把旧 module 的权重张量赋值给新 module 的 weight
                model._modules[name].weight = old_module.weight
                # 若有 bias
                if module.bias is not None:
                    model._modules[name].bias = old_module.bias

            # 如果指定了 post_processing_function，就取旧 module 上是否有这个函数
            # 如果有，就调用它。这样可以让一些需要额外初始化／配置的操作在替换后执行。
            if post_processing_function is not None:
                func = getattr(module, post_processing_function, None)
                if func is not None:
                    func(module)
    return model
