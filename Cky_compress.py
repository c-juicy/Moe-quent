import torch
from transformers import AutoModelForCausalLM
from Cky.model.hf.deepseekv2 import DeepSeekV2MoEMiLo as AutoMiLoHFModel
from Cky.quentizers.quentizer import *


def main():
    device = "cuda"
    save_dir = "/remote-home/share/cky/DeepSeek_V2_Cky_s1"
    model_dir = "/remote-home/share/cky/DeepSeek-V2-Lite/model_dir/"
    compress_config = BaseCompressConfig(
                                        # quantization config
                                         nbits = 8, 
                                         group_size = 64, 
                                         quant_scale = False, 
                                         quant_zero = False, 
                                         axis = 1,
                                        # compensator config
                                         iter = 10,
                                         sparse_rank = 16,
                                         dense_rank = 512,
                                         rank_strategy = None,
                                         compensator_dtype  = "int3"
                                         ) 
    model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    AutoMiLoHFModel.compress_model(model, 
                                   compress_config=compress_config, 
                                   device=device)   
    AutoMiLoHFModel.save_compressed(model, save_dir)




if __name__ == "__main__":
    main()

