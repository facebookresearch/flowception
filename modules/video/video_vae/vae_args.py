vae_args_sd3 = {
#   "_class_name": "CausalVideoVAE",
#   "_diffusers_version": "0.29.2",
  "add_post_quant_conv": True,
  "decoder_act_fn": "silu",
  "decoder_block_dropout": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "decoder_block_out_channels": [
    128,
    256,
    512,
    512
  ],
  "decoder_in_channels": 16,
  "decoder_layers_per_block": [
    3,
    3,
    3,
    3
  ],
  "decoder_norm_num_groups": 32,
  "decoder_out_channels": 3,
  "decoder_spatial_up_sample": [
    True,
    True,
    True,
    False
  ],
  "decoder_temporal_up_sample": [
    True,
    True,
    True,
    False
  ],
  "decoder_type": "causal_vae_conv",
  "decoder_up_block_types": [
    "UpDecoderBlockCausal3D",
    "UpDecoderBlockCausal3D",
    "UpDecoderBlockCausal3D",
    "UpDecoderBlockCausal3D"
  ],
  "downsample_scale": 8,
  "encoder_act_fn": "silu",
  "encoder_block_dropout": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "encoder_block_out_channels": [
    128,
    256,
    512,
    512
  ],
  "encoder_double_z": True,
  "encoder_down_block_types": [
    "DownEncoderBlockCausal3D",
    "DownEncoderBlockCausal3D",
    "DownEncoderBlockCausal3D",
    "DownEncoderBlockCausal3D"
  ],
  "encoder_in_channels": 3,
  "encoder_layers_per_block": [
    2,
    2,
    2,
    2
  ],
  "encoder_norm_num_groups": 32,
  "encoder_out_channels": 16,
  "encoder_spatial_down_sample": [
    True,
    True,
    True,
    False
  ],
  "encoder_temporal_down_sample": [
    True,
    True,
    True,
    False
  ],
  "encoder_type": "causal_vae_conv",
  "interpolate": False,
  "sample_size": 256,
  "scaling_factor": 0.13025
}