from jax import numpy as jnp

from . import Transformer, ViT, TNT, CaiT, MLPMixer


def create_trunk(model_name: str,
                 output_dim: int = 1000,
                 dtype: jnp.dtype = jnp.float32):

    if model_name == 'txt_b':
        return Transformer(output_dim=output_dim,
                   num_layers=12,
                   num_heads=12,
                   embed_dim=768,
                   rotary_qk=True,
                   dtype=dtype)    
    elif model_name == 'spec_b':
        return TNT(output_dim=output_dim,
                   num_layers=12,
                   inner_num_heads=4,
                   outer_num_heads=6,
                   inner_embed_dim=24,
                   outer_embed_dim=384,
                   patch_shape=(2, 80),
                   transformed_patch_shape=(1, 80),
                   rotary_qk=True)
    elif model_name == 'vit_b_patch32':
        return ViT(output_dim=output_dim,
                   num_layers=12,
                   num_heads=12,
                   embed_dim=768,
                   patch_shape=(32, 32),
                   dtype=dtype)
    elif model_name == 'vit_b_patch16':
        return ViT(output_dim=output_dim,
                   num_layers=12,
                   num_heads=12,
                   embed_dim=768,
                   patch_shape=(16, 16),
                   dtype=dtype)
    elif model_name == 'vit_l_patch32':
        return ViT(output_dim=output_dim,
                   num_layers=24,
                   num_heads=16,
                   embed_dim=1024,
                   patch_shape=(32, 32),
                   dtype=dtype)
    elif model_name == 'vit_l_patch16':
        return ViT(output_dim=output_dim,
                   num_layers=24,
                   num_heads=16,
                   embed_dim=1024,
                   patch_shape=(16, 16),
                   dtype=dtype)
    elif model_name == 'tnt_s_patch16':
        return TNT(output_dim=output_dim,
                   num_layers=12,
                   inner_num_heads=4,
                   outer_num_heads=10,
                   inner_embed_dim=40,
                   outer_embed_dim=640,
                   patch_shape=(16, 16),
                   transformed_patch_shape=(4, 4))
    elif model_name == 'tnt_b_patch16':
        return TNT(output_dim=output_dim,
                   num_layers=12,
                   inner_num_heads=4,
                   outer_num_heads=6,
                   inner_embed_dim=24,
                   outer_embed_dim=384,
                   patch_shape=(16, 16),
                   transformed_patch_shape=(4, 4))
    elif model_name == 'cait_xxs_24':
        return CaiT(output_dim=output_dim,
                    num_layers=24,
                    num_layers_token_only=2,
                    num_heads=4,
                    embed_dim=192,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.05,
                    layerscale_eps=1e-5)
    elif model_name == 'cait_xxs_36':
        return CaiT(output_dim=output_dim,
                    num_layers=36,
                    num_layers_token_only=2,
                    num_heads=4,
                    embed_dim=192,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.1,
                    layerscale_eps=1e-6)
    elif model_name == 'cait_xs_24':
        return CaiT(output_dim=output_dim,
                    num_layers=24,
                    num_layers_token_only=2,
                    num_heads=6,
                    embed_dim=288,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.05,
                    layerscale_eps=1e-5)
    elif model_name == 'cait_xs_36':
        return CaiT(output_dim=output_dim,
                    num_layers=36,
                    num_layers_token_only=2,
                    num_heads=6,
                    embed_dim=288,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.1,
                    layerscale_eps=1e-6)
    elif model_name == 'cait_s_24':
        return CaiT(output_dim=output_dim,
                    num_layers=24,
                    num_layers_token_only=2,
                    num_heads=8,
                    embed_dim=384,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.1,
                    layerscale_eps=1e-6)
    elif model_name == 'cait_s_36':
        return CaiT(output_dim=output_dim,
                    num_layers=36,
                    num_layers_token_only=2,
                    num_heads=8,
                    embed_dim=384,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.2,
                    layerscale_eps=1e-6)
    elif model_name == 'cait_s_48':
        return CaiT(output_dim=output_dim,
                    num_layers=48,
                    num_layers_token_only=2,
                    num_heads=8,
                    embed_dim=384,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.3,
                    layerscale_eps=1e-6)
    elif model_name == 'cait_m_24':
        return CaiT(output_dim=output_dim,
                    num_layers=24,
                    num_layers_token_only=2,
                    num_heads=16,
                    embed_dim=768,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.2,
                    layerscale_eps=1e-5)
    elif model_name == 'cait_m_36':
        return CaiT(output_dim=output_dim,
                    num_layers=36,
                    num_layers_token_only=2,
                    num_heads=16,
                    embed_dim=768,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.3,
                    layerscale_eps=1e-6)
    elif model_name == 'cait_m_48':
        return CaiT(output_dim=output_dim,
                    num_layers=48,
                    num_layers_token_only=2,
                    num_heads=16,
                    embed_dim=768,
                    patch_shape=(16, 16),
                    stoch_depth_rate=0.4,
                    layerscale_eps=1e-6)
    elif model_name == 'mixer_s_patch32':
        return MLPMixer(output_dim=output_dim,
                        num_layers=8,
                        embed_dim=512,
                        patch_shape=(32, 32))
    elif model_name == 'mixer_s_patch16':
        return MLPMixer(output_dim=output_dim,
                        num_layers=8,
                        embed_dim=512,
                        patch_shape=(16, 16))
    elif model_name == 'mixer_b_patch32':
        return MLPMixer(output_dim=output_dim,
                        num_layers=12,
                        embed_dim=768,
                        patch_shape=(32, 32))
    elif model_name == 'mixer_s_patch32':
        return MLPMixer(output_dim=output_dim,
                        num_layers=12,
                        embed_dim=768,
                        patch_shape=(16, 16))
    elif model_name == 'mixer_l_patch32':
        return MLPMixer(output_dim=output_dim,
                        num_layers=24,
                        embed_dim=1024,
                        patch_shape=(32, 32))
    elif model_name == 'mixer_l_patch16':
        return MLPMixer(output_dim=output_dim,
                        num_layers=32,
                        embed_dim=1024,
                        patch_shape=(16, 16))
    else:
        raise RuntimeError('Model not found.')