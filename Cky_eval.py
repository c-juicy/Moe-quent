from Cky.quentizers.quentizer import *
from Cky.model.hf.deepseekv2 import DeepSeekV2MoEMiLo as AutoMiLoHFModel
# from MiLo.models.hf.deepseek import DeepSeekMoEMiLo
from Cky.engine.hf import AutoTokenizer
from evaluation.eval_wikitext2_ppl import eval_wikitext2_perplexity
from evaluation.eval_fewshots import eval_fewshots
from evaluation.eval_zeroshot import eval_zeroshot

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():

    orig_model_dir = "/remote-home/share/cky/DeepSeek-V2-Lite/model_dir/"
    quant_model_dir = "/remote-home/share/cky/DeepSeek_V2_Cky_s1"
    model_id = "deepseek-ai/DeepSeek-V2-Lite" 
    model = AutoMiLoHFModel.from_compressed(quant_model_dir)
    tokenizer  = AutoTokenizer.from_pretrained(orig_model_dir,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token =tokenizer.eos_token
    eval_zeroshot(model,tokenizer,quant_model_dir)
    # eval_wikitext2_perplexity(model,tokenizer,quant_model_dir)
    # eval_fewshots(model,tokenizer,quant_model_dir)
    return

if __name__ == "__main__":
    main()
