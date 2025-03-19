# paper3_code

2025/3/16 update MLLA + MambaVision for encoder-decoder model with new skip connection

MLLA-T(c=[64,128,256,512]) + MambaVision(d=[1,1,1,1]) + new skip

**d1=0.9676, abs_rel=0.0556**

add part Search Space (mlp_ratio, num_heads)

# TODO
- Search Space (mamba)
- SuperNet Training (Knowledge Distillation)
- NAS Search
- Retrain