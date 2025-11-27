import torch
import copy
from torch import uint8, int32, float16, nn, Tensor
from enum import Enum
from Cky.utils.utils import is_divisible, encode_safetensor_type, decode_safetensor_type
from Cky.logs.logger import init_logger
from typing import Union    
from .bitpack import BitPack

logger = init_logger(__name__)


_META_TYPE = {
    "scale": torch.Tensor,
    "zero": torch.Tensor,
    "zero_scale": torch.Tensor,
    "compute_dtype": torch.dtype,
    "quant_zero": bool,
    "quant_scale": bool,
    "view_as_float": bool,
    "unpack_view_dtype": torch.dtype,
    "packing": str,
    "axis": int,
    "group_size": int,
    "nbits": int,
    "shape": torch.Size,
    "channel_wise": bool,
    "optimize": bool,
    "round_zero": bool,

    "ranks": dict,
    "compensator_dtype": str,
    "compensator_quant_gs": int,
}


class Quantizer:
    SUPPORTED_BITS = [8]
    
    bit_to_packing = {
        8: "8bit_u8",
        4: "4bit_u8",
        3: "3bit_32",
    }

    pack = {
        "8bit_u8": BitPack.pack_8bit_u8,
        "4bit_u8": BitPack.pack_4bit_u8,
        "3bit_32": BitPack.pack_3bit_32,
    }

    unpack = {
        "8bit_u8": BitPack.unpack_8bit_u8,
        "4bit_u8": BitPack.unpack_4bit_u8,
        "3bit_32": BitPack.unpack_3bit_32,
    }

    unpack_view_dtype = {
        "8bit_u8": uint8,
        "4bit_u8": uint8,
        "3bit_32": int32,
    }
    @classmethod
    def quantize(
        cls,
        tensor: Tensor,
        nbits: float = 8,
        channel_wise: bool = True,
        group_size: int = 64,
        axis: int = 0,
        
    ):
        pass

    @classmethod
    def base_quantize(
        cls,
        tensor: Tensor,
        nbits: float = 8,
        channel_wise: bool = True,
        bitpack: bool = True,
        group_size: int = 64,
        optimize: bool = False,
        round_zero: bool = False,
        axis: int = 0,
        compute_dtype: Union[torch.dtype, None] = None,
        view_as_float: bool = False,
        device: str = "cuda",
    ):
        assert tensor.dim() == 2
        assert nbits in Quantizer.SUPPORTED_BITS, (
            "nbits=" + str(nbits) + " not supported."
        )
        assert axis in [0, 1], "axis should be either 0 or 1"
        if group_size is not None:
            assert is_divisible(tensor.numel(), group_size), (
                "group_size should be divisble by the total tensor dimensions. shape: "
                + str(tensor.shape)
                + ", group_size: "
                + str(group_size)
            )
            
        W = tensor.float()
        shape = W.shape
        logger.info(f"[Base quantize]Tensor shape is {shape}")
        
        # Get min/max values
        if not channel_wise:
            _min, _max = W.min(), W.max()
            optimize = False
        else:
            _min = W.min(axis=axis, keepdim=True)[0]
            _max = W.max(axis=axis, keepdim=True)[0]
        
        max_int = round(2**nbits - 1)
        min_int = 0
        min_max = [min_int, max_int]
        # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
        scale = (max_int / (_max - _min)).clamp(
            max=2e4
        )  # clamp to avoid half-precision problems
        zero = -_min * scale
        
        W_q = torch.round(W * scale + zero).clamp(min_max[0], min_max[1])
        
        # Store meta-data (we invert the scale for dequantization)
        meta = {
            "nbits": nbits,
            "group_size": group_size,
            "shape": shape,
            "scale": 1.0 / scale,
            "zero": zero,
            "axis": axis,
            "packing": Quantizer.bit_to_packing[nbits],
        }
        meta["unpack_view_dtype"] = Quantizer.unpack_view_dtype[meta["packing"]]
        # Pack bits
        meta["view_as_float"] = view_as_float
        if bitpack:
            W_q = Quantizer.pack[meta["packing"]](W_q)
            if view_as_float:
                W_q = W_q.view(
                    torch.float32 if compute_dtype is None else compute_dtype
                )  # store quantized weights as compute_dtype
        else:
            W_q = W_q.to(tensor.dtype)
            meta["packing"] = None
        
        # cleanup
        del W, _min, _max
        torch.cuda.empty_cache()

        return W_q, meta
            
    @classmethod
    def dequantize(cls, W_q: Tensor, meta: dict) -> Tensor:
        compute_dtype = meta["compute_dtype"] if ("compute_dtype" in meta) else float16
        if meta["packing"]:
            if meta["view_as_float"]:
                W_q = W_q.view(meta["unpack_view_dtype"])
            W_r = Quantizer.unpack[meta["packing"]](W_q, dtype=compute_dtype)
            if meta["nbits"] == 3:
                W_r = W_r[
                    : meta["group_size"]
                    if meta["axis"] == 0
                    else meta["shape"][0] * meta["shape"][1] // meta["group_size"]
                ]
        else:
            W_r = W_q.to(compute_dtype)
        W_r = ((W_r - meta["zero"]) * meta["scale"]).reshape(meta["shape"])
        return W_r
    
    @classmethod
    def to_inplace(cls, W_q: Tensor, meta: dict, device) -> tuple:
        compute_dtype = meta["compute_dtype"] if ("compute_dtype" in meta) else float16
        if W_q is not None:
            W_q = W_q.to(device).contiguous()
        for key in meta:
            if type(meta[key]) == torch.Tensor:
                meta[key] = (
                    (
                        meta[key].to(compute_dtype)
                        if torch.is_floating_point(meta[key])
                        else meta[key]
                    )
                    .to(device)
                    .contiguous()
                )
        return W_q, meta
    
    @classmethod
    def to_ooplace(cls, W_q: Tensor, meta: dict, device) -> tuple:
        compute_dtype = meta["compute_dtype"] if ("compute_dtype" in meta) else float16
        if W_q is not None:
            W_q_c = W_q.to(device).contiguous()
        else:
            W_q_c = None
        meta_c = {}
        for key in meta:
            if type(meta[key]) == torch.Tensor:
                meta_c[key] = (
                    (
                        meta[key].to(compute_dtype)
                        if torch.is_floating_point(meta[key])
                        else meta[key]
                    )
                    .to(device)
                    .contiguous()
                )
            else:
                meta_c[key] = meta[key]
        return W_q_c, meta_c
    
    @classmethod
    def cuda(cls, W_q: Tensor, meta: dict, device) -> tuple:
        return Quantizer.to_inplace(W_q, meta, device=device)

    @classmethod
    def cpu(cls, W_q: Tensor, meta: dict) -> tuple:
        return Quantizer.to_ooplace(W_q, meta, device="cpu")

# import copy
# from typing import Optional, Union, TypeVar
# from enum import Enum
# from torch import uint8, int32, float16, nn, Tensor
# from logs.logger import init_logger
# from quentizers.quentizer import Quantizer
# import torch

logger = init_logger(__name__)

class CkyBackend(Enum):
    # Name of the forward functions
    PYTORCH = "forward_pytorch"
    PYTORCH_COMPILE = "forward_pytorch_compile"


# Main linear layer
class CkyLinear(nn.Module):
    backend = CkyBackend.PYTORCH
    
    def __init__(
        self,
        linear_layer: Union[nn.Module, None],
        compress_config: dict,
        compute_dtype: torch.dtype = float16,
        device: str = 'cuda',
        del_orig: bool = True,
        need_initialize: bool = True,
    ):
        super().__init__()
        
        self.ready = False
        self.in_gpu = False
        self.bias = None
        self.device = device
        self.linear_layer = linear_layer
        self.compress_config = copy.deepcopy(compress_config)
        self.W_q = None # 量化权重
        self.meta = None    # 量化的信息
        self.compute_dtype = compute_dtype
        self.encoded_state_dict = (
            True  # This makes state_dict compatible with safetensors
        )
        self.del_orig = del_orig
        self.offload_meta = (
            self.compress_config.pop("offload_meta")
            if (self.compress_config is not None)
            else None
        )
        
        self.set_backend(CkyLinear.backend)
        if linear_layer is not None:
            self.orig_shape = self.linear_layer.weight.data.shape
            self.name = linear_layer.name

        
        if need_initialize:
            self.initialize()
    
    def initialize(self):
        if self.linear_layer is not None:
            self.compress(self.linear_layer.weight.data, **self.compress_config)
            self.bias = (
                None
                if (self.linear_layer.bias is None)
                else self.linear_layer.bias.to(
                    device=self.device, dtype=self.compute_dtype
                )
            )
        if self.del_orig:
            del self.linear_layer
        torch.cuda.empty_cache()
    
    # Set backends
    @classmethod
    def set_backend(cls, backend: CkyBackend):
        CkyLinear.backend = backend
        cls.forward = getattr(cls, backend.value)
    
    
    def cuda(self, device):
        self.meta["compute_dtype"] = self.compute_dtype
        if type(self.W_q) == nn.parameter.Parameter:
            self.W_q.data, self.meta = Quantizer.cuda(self.W_q.data, self.meta, device)
        else:
            self.W_q, self.meta = Quantizer.cuda(self.W_q, self.meta, device)
        if self.meta["quant_zero"]:
            if "zero_q" in self.meta:
                self.meta["zero_q"], self.meta["meta_zero"] = Quantizer.cuda(
                    self.meta["zero_q"], self.meta["meta_zero"], device
                )
            else:
                _, self.meta["meta_zero"] = Quantizer.cuda(
                    None, self.meta["meta_zero"], device
                )
        elif "zero" in self.meta:
            self.meta["zero"] = self.meta["zero"].to(device)

        if self.meta["quant_scale"]:
            if "scale_q" in self.meta:
                self.meta["scale_q"], self.meta["meta_scale"] = Quantizer.cuda(
                    self.meta["scale_q"], self.meta["meta_scale"], device
                )
            else:
                _, self.meta["meta_scale"] = Quantizer.cuda(
                    None, self.meta["meta_scale"], device
                )
        elif "scale" in self.meta:
            self.meta["scale"] = self.meta["scale"].to(device)

        # #Use zero/scale with streams for dequantization is faster than packing in "zero_scale"
        # for key in ["zero", "zero_q", "scale", "scale_q"]:
        #     if((key in self.meta) and self.offload_meta):
        #         self.meta[key] = self.meta[key].contiguous().cpu().pin_memory()

        if self.offload_meta:
            if "zero_scale" not in self.meta:
                if self.meta["quant_scale"] and self.meta["quant_zero"]:
                    self.meta["zero_scale"] = torch.stack(
                        (self.meta["zero_q"], self.meta["scale_q"])
                    )
                    del self.meta["scale_q"], self.meta["zero_q"]
                else:
                    self.meta["zero_scale"] = torch.stack(
                        (self.meta["zero"], self.meta["scale"])
                    ).to(self.compute_dtype)
                    del self.meta["scale"], self.meta["zero"]

            self.meta["zero_scale"] = (
                self.meta["zero_scale"].contiguous().cpu().pin_memory()
            )

        if self.bias is not None:
            self.bias = self.bias.to(device=device, dtype=self.compute_dtype)

        self.W_q = nn.Parameter(self.W_q, requires_grad=False)
        self.device = device
        self.in_gpu = True

        torch.cuda.empty_cache()

        return self
    
    def to(self, *args, **kwargs):
        # TODO: later
        return self

    def type(self, dst_type):
        # TODO: later
        return self

    def half(self, *args, **kwargs):
        return self

    def bfloat16(self, *args, **kwargs):
        # TODO: later
        return self

    def float(self, *args, **kwargs):
        # TODO: later
        return self

    def double(self, *args, **kwargs):
        return self

    def cpu(self):
        # TODO: later
        return self
    
    # state_dict is encoded by default for safetensors support. You can get the raw dict by setting self.encoded_state_dict=False. \
    # Note: you can't change the state once it's done
    def state_dict(self, *args, **kwargs):  # nn.Module override compatible
        if (
            self.compress_config["scale_quant_params"]
            or self.compress_config["zero_quant_params"]
        ) and self.encoded_state_dict:
            raise Exception(
                "Unsupported serialization for quantized scale/zero and self.encoded_state_dict=True"
            )
            # TODO: add support for quantized zero/scale case (quant_config and zero/scale)

        _encode_type = (
            encode_safetensor_type if (self.encoded_state_dict) else lambda z: z
        )

        # Core data
        state = {"W_q": self.W_q} | {k: _encode_type(v) for k, v in self.meta.items()}
        if self.bias is not None:
            state["bias"] = self.bias
        state["offload_meta"] = _encode_type(self.offload_meta)

        # Encoding flag
        if self.encoded_state_dict:
            state["encoded_state_dict"] = _encode_type(self.encoded_state_dict)

        # Quant config
        state["stores_quant_config"] = _encode_type(True)
        for k in self.compress_config["weight_quant_params"]:
            state[k] = _encode_type(self.compress_config["weight_quant_params"][k])

        if "destination" in kwargs and "prefix" in kwargs:
            for key, value in state.items():
                kwargs["destination"][kwargs["prefix"] + key] = value

        # compensator config
        self.compress_config["compensator_params"].pop("ranks",None)
        for k in self.compress_config["compensator_params"]:
            state[k] = _encode_type(self.compress_config["compensator_params"][k])

        return state


    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "encoded_state_dict" in state_dict:
            encoded_state_dict = True
            state_dict.pop("encoded_state_dict")
        else:
            encoded_state_dict = False

        _decode_type = (
            decode_safetensor_type if (encoded_state_dict) else lambda z, w: z
        )

        # Quant-config
        if state_dict.pop(
            "stores_quant_config", False
        ):  # check for backward compatibility
            self.compress_config = {
                "weight_quant_params": {
                    k: _decode_type(state_dict[k], _META_TYPE[k])
                    for k in [
                        "nbits",
                        "channel_wise",
                        "group_size",
                        "optimize",
                        "round_zero",
                        "axis",
                        "view_as_float",
                    ]
                }
            }
            # TODO: scale/zero quant use-case
            self.compress_config["scale_quant_params"] = state_dict.pop(
                "scale_quant_params", None
            )
            self.compress_config["zero_quant_params"] = state_dict.pop(
                "zero_quant_params", None
            )
        self.compress_config["compensator_params"] = {}
        self.compress_config["compensator_params"]["sparse_rank"] = state_dict.pop(
                "sparse_rank", None
            )
        self.compress_config["compensator_params"]["iter"] = state_dict.pop(
                "iter", None
            )
        self.compress_config["compensator_params"]["dense_rank"] = state_dict.pop(
                "dense_rank", None
            )
        self.compress_config["compensator_params"]["rank_strategy"] = state_dict.pop(
                "rank_strategy", None
            )
        self.compress_config["compensator_params"]["compensator_dtype"] = state_dict.pop(
                "compensator_dtype", None
            )
        self.compress_config["compensator_params"]["compensator_quant_gs"] = state_dict.pop(
                "compensator_quant_gs", None
            )
            
        # W_q/ bias
        self.W_q = state_dict.pop("W_q")
        self.bias = state_dict.pop("bias", None)

        # Meta
        self.offload_meta = _decode_type(state_dict.pop("offload_meta", False), bool)
        if "meta" in state_dict:
            self.meta = state_dict["meta"]  # Backward compatibility
        else:
            self.meta = {
                k: _decode_type(v, _META_TYPE[k]) for k, v in state_dict.items()
            }  # safetensors version

        # Meta-data offloading
        if self.offload_meta is None:
            self.offload_meta = False
        for key in ["zero", "zero_q", "scale", "scale_q", "zero_scale"]:
            if key in self.meta and self.offload_meta:
                self.meta[key] = self.meta[key].cpu().contiguous().pin_memory()

        # Float view settings
        if "unpack_view_dtype" not in self.meta:
            self.meta["unpack_view_dtype"] = Quantizer.unpack_view_dtype[
                self.meta["packing"]
            ]

        if "view_as_float" not in self.meta:
            self.meta["view_as_float"] = False

        if "meta_scale" in self.meta:
            if "view_as_float" not in self.meta["meta_scale"]:
                self.meta["meta_scale"]["view_as_float"] = False

        if "meta_zero" in self.meta:
            if "view_as_float" not in self.meta["meta_zero"]:
                self.meta["meta_zero"]["view_as_float"] = False

        # Check GPU
        self.cuda(self.device)
        self.ready = True

        # Set in_features/out_features
        self.in_features, self.out_features = self.meta["shape"][::-1]
    
    def compress(
        self,
        W: Tensor,
        weight_quant_params: dict,
        scale_quant_params: dict,
        zero_quant_params: dict,
        compensator_params: dict,
    ) -> None:
        
        quant_scale = scale_quant_params is not None
        quant_zero = zero_quant_params is not None
        
        self.in_features, self.out_features = W.t().shape
        W_unquant = W.to(self.device)
        W_q = None
        
        # Quantize
        logger.info(f"quantize {self.name} to {weight_quant_params['nbits']} bits")
        W_q, meta = Quantizer.base_quantize(
            W,
            device=self.device,
            compute_dtype=self.compute_dtype,
            **weight_quant_params,
        )
        meta.update({"quant_scale": quant_scale, "quant_zero": quant_zero})
        
        W_q_dequant = Quantizer.dequantize(W_q, meta).to(self.device)
        Error = W_unquant - W_q_dequant
        # logger.info(f"norm={torch.norm(Error,p='fro')}")
        
        if meta["quant_zero"]:
            meta["zero_q"], meta["meta_zero"] = Quantizer.base_quantize(
                meta["zero"],
                device=self.device,
                view_as_float=False,
                **zero_quant_params,
            )
            del meta["zero"]
            meta["meta_zero"]["compute_dtype"] = self.compute_dtype

        if meta["quant_scale"]:
            meta["scale_q"], meta["meta_scale"] = Quantizer.base_quantize(
                meta["scale"],
                device=self.device,
                view_as_float=False,
                **scale_quant_params,
            )
            del meta["scale"]
            meta["meta_scale"]["compute_dtype"] = self.compute_dtype

        
        self.W_q = W_q
        self.meta = meta
        self.cuda(self.device)
        self.ready = True
    
    
    def dequantize(self):
        assert self.ready, "model was not quantized"
        W_q, meta = self.W_q, self.meta
        device = W_q.device
        del_keys = set()
        
        if meta["quant_zero"]:
            meta["zero"] = Quantizer.dequantize(
                meta["zero_q"].to(device=device), meta["meta_zero"]
            )
            del_keys.add("zero")

        if meta["quant_scale"]:
            meta["scale"] = Quantizer.dequantize(
                meta["scale_q"].to(device=device), meta["meta_scale"]
            )
            del_keys.add("scale")
        
        W_est = Quantizer.dequantize(W_q, meta)
        
        # Cleanup
        for key in del_keys:
            del meta[key]
        return W_est

    def matmul(self, x: Tensor, transpose: bool = True) -> Tensor:
        weight = self.dequantize()
        return torch.matmul(x, weight.t() if (transpose) else weight)
    
    def forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        logger.info("forward_pytorch")
        out = torch.matmul(x, self.dequantize().t())
        if self.bias is not None:
            out += self.bias
        return out

    @torch.compile()
    def forward_pytorch_compile(self, x: torch.Tensor) -> torch.Tensor:
        logger.info("forward_pytorch_compile")
        return self.forward_pytorch(x)
    
    


def milo_base_compress_config(
    #quantization config
    nbits: int = 4,
    group_size: int = 64,
    quant_zero: bool = True,
    quant_scale: bool = False,
    offload_meta: bool = False,  # meta-data should be quantized with the same settings to use offload_meta
    view_as_float: bool = False,
    axis: int = 0,
    #compensator config
    iter: int = 10,
    sparse_rank: int = 0,
    dense_rank: int = 0,
    rank_strategy: str = None,
    compensator_dtype: str = "int3",
    compensator_quant_gs: int = 64,
):
    assert (
        nbits in Quantizer.SUPPORTED_BITS
    ), "nbits value not supported. Check Quantizer.SUPPORTED_BITS."

    if group_size is not None:
        assert is_divisible(
            group_size, 8
        ), "Invalid group_size param: the value should be a multiple of 8."
    weight_quant_params = {
        "nbits": nbits,
        "channel_wise": True,
        "group_size": group_size,
        "optimize": True,
        "round_zero": True if nbits == 4 else False,
        "axis": axis,
        "view_as_float": view_as_float,
    }

    if offload_meta:
        if quant_scale != quant_zero:
            # print(colored("quant_zero and quant_scale must be the same when offload_meta is set to True. Setting quant_scale=quant_zero." , 'yellow'))
            quant_scale = quant_zero

        scale_quant_params = (
            {"nbits": 8, "channel_wise": True, "group_size": 128, "optimize": False}
            if (quant_scale)
            else None
        )
        zero_quant_params = (
            {"nbits": 8, "channel_wise": True, "group_size": 128, "optimize": False}
            if (quant_zero)
            else None
        )

    else:
        scale_quant_params = (
            {"nbits": 8, "channel_wise": True, "group_size": 128, "optimize": False}
            if (quant_scale)
            else None
        )
        zero_quant_params = (
            {"nbits": 8, "channel_wise": False, "group_size": None, "optimize": False}
            if (quant_zero)
            else None
        )

    compensator_params = {
        "iter": iter,
        "sparse_rank": sparse_rank,
        "dense_rank": dense_rank,
        "rank_strategy": rank_strategy,
        "compensator_dtype": compensator_dtype,
        "compensator_quant_gs": compensator_quant_gs
    }

    return {
        # quantization configs
        "weight_quant_params": weight_quant_params,
        "scale_quant_params": scale_quant_params,
        "zero_quant_params": zero_quant_params,
        "offload_meta": offload_meta,
        # compensator configs
        "compensator_params": compensator_params,
    }



# Alias: follow similar Auto-GPTQ naming
BaseCompressConfig = milo_base_compress_config