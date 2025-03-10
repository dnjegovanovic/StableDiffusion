import torch


def load_from_standard_weights(input_file: str, device: str) -> dict[str, torch.Tensor]:
    # Taken from: https://github.com/kjsman/stable-diffusion-pytorch/issues/7#issuecomment-1426839447
    original_model = torch.load(input_file, map_location=device, weights_only=False)[
        "state_dict"
    ]

    converted = {}
    converted["diffusion"] = {}
    converted["encoder"] = {}
    converted["decoder"] = {}
    converted["clip"] = {}

    converted["diffusion"]["temporal_embedding.linear_1.weight"] = original_model[
        "model.diffusion_model.time_embed.0.weight"
    ]
    converted["diffusion"]["temporal_embedding.linear_1.bias"] = original_model[
        "model.diffusion_model.time_embed.0.bias"
    ]
    converted["diffusion"]["temporal_embedding.linear_2.weight"] = original_model[
        "model.diffusion_model.time_embed.2.weight"
    ]
    converted["diffusion"]["temporal_embedding.linear_2.bias"] = original_model[
        "model.diffusion_model.time_embed.2.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.0.0.weight"] = original_model[
        "model.diffusion_model.input_blocks.0.0.weight"
    ]
    converted["diffusion"]["denoising_unet.encoders.0.0.bias"] = original_model[
        "model.diffusion_model.input_blocks.0.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.1.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.1.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.1.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.1.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.1.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.input_blocks.1.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.input_blocks.1.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.1.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.1.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.1.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.1.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.1.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.1.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.input_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.1.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.input_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.1.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.encoders.1.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.1.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.1.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.encoders.1.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.1.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.1.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.1.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.1.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.1.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.output_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.1.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.1.1.output_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.1.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.2.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.2.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.2.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.2.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.input_blocks.2.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.input_blocks.2.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.2.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.2.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.2.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.2.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.2.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.2.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.input_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.2.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.input_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.2.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.encoders.2.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.2.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.2.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.encoders.2.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.2.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.2.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.2.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.2.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.2.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.output_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.2.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.2.1.output_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.2.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.3.0.weight"] = original_model[
        "model.diffusion_model.input_blocks.3.0.op.weight"
    ]
    converted["diffusion"]["denoising_unet.encoders.3.0.bias"] = original_model[
        "model.diffusion_model.input_blocks.3.0.op.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.4.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.input_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.input_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.encoders.4.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.4.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.4.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.encoders.4.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.4.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.4.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.4.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.4.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.4.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.output_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.4.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.4.1.output_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.4.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.5.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.5.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.5.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.5.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.input_blocks.5.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.input_blocks.5.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.5.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.5.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.5.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.5.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.5.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.5.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.input_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.5.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.input_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.5.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.encoders.5.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.5.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.5.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.encoders.5.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.5.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.5.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.5.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.5.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.5.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.output_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.5.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.5.1.output_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.5.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.6.0.weight"] = original_model[
        "model.diffusion_model.input_blocks.6.0.op.weight"
    ]
    converted["diffusion"]["denoising_unet.encoders.6.0.bias"] = original_model[
        "model.diffusion_model.input_blocks.6.0.op.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.7.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.input_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.input_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.encoders.7.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.7.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.7.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.encoders.7.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.7.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.7.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.7.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.7.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.7.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.output_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.7.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.7.1.output_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.7.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.8.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.8.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.8.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.8.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.input_blocks.8.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.input_blocks.8.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.8.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.8.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.8.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.8.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.8.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.8.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.input_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.8.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.input_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.8.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.encoders.8.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.8.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.8.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.encoders.8.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.8.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.8.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.8.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.8.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.8.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.output_projection.weight"] = (
        original_model["model.diffusion_model.input_blocks.8.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.8.1.output_projection.bias"] = (
        original_model["model.diffusion_model.input_blocks.8.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.9.0.weight"] = original_model[
        "model.diffusion_model.input_blocks.9.0.op.weight"
    ]
    converted["diffusion"]["denoising_unet.encoders.9.0.bias"] = original_model[
        "model.diffusion_model.input_blocks.9.0.op.bias"
    ]
    converted["diffusion"]["denoising_unet.encoders.10.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.10.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.10.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.10.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.10.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.input_blocks.10.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.input_blocks.10.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.10.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.10.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.10.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.10.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.10.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.11.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.11.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.11.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.11.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.input_blocks.11.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.input_blocks.11.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.input_blocks.11.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.input_blocks.11.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.input_blocks.11.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.encoders.11.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.input_blocks.11.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.middle_block.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.middle_block.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.middle_block.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.middle_block.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.middle_block.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.middle_block.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.middle_block.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.middle_block.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.middle_block.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.middle_block.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.middle_block.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.middle_block.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.input_projection.weight"] = (
        original_model["model.diffusion_model.middle_block.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.input_projection.bias"] = (
        original_model["model.diffusion_model.middle_block.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.bottleneck.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.bottleneck.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.bottleneck.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.bottleneck.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.bottleneck.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.bottleneck.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.bottleneck.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.bottleneck.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.bottleneck.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.output_projection.weight"] = (
        original_model["model.diffusion_model.middle_block.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.1.output_projection.bias"] = (
        original_model["model.diffusion_model.middle_block.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.middle_block.2.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.middle_block.2.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.feature_conv.weight"] = (
        original_model["model.diffusion_model.middle_block.2.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.feature_conv.bias"] = (
        original_model["model.diffusion_model.middle_block.2.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.temporal_linear.weight"] = (
        original_model["model.diffusion_model.middle_block.2.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.temporal_linear.bias"] = (
        original_model["model.diffusion_model.middle_block.2.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.middle_block.2.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.middle_block.2.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.merged_conv.weight"] = (
        original_model["model.diffusion_model.middle_block.2.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.bottleneck.2.merged_conv.bias"] = (
        original_model["model.diffusion_model.middle_block.2.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.0.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.0.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.0.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.0.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.0.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.0.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.0.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.0.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.0.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.0.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.0.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.0.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.0.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.1.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.1.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.1.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.1.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.1.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.1.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.1.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.1.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.1.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.1.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.1.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.1.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.1.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.2.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.2.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.2.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.2.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.2.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.2.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.2.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.2.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.2.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.2.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.2.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.2.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.2.1.conv.weight"] = original_model[
        "model.diffusion_model.output_blocks.2.1.conv.weight"
    ]
    converted["diffusion"]["denoising_unet.decoders.2.1.conv.bias"] = original_model[
        "model.diffusion_model.output_blocks.2.1.conv.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.3.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.3.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.3.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.3.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.3.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.3.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.3.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.3.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.3.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.3.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.3.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.3.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.3.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.4.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.4.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.4.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.4.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.4.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.4.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.4.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.4.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.4.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.4.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.4.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.4.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.5.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.5.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.5.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.5.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.5.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.5.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.5.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.5.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.5.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.5.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.5.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.5.2.conv.weight"] = original_model[
        "model.diffusion_model.output_blocks.5.2.conv.weight"
    ]
    converted["diffusion"]["denoising_unet.decoders.5.2.conv.bias"] = original_model[
        "model.diffusion_model.output_blocks.5.2.conv.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.6.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.6.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.6.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.6.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.6.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.6.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.6.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.6.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.6.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.6.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.6.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.6.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.6.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.7.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.7.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.7.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.7.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.7.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.7.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.7.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.7.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.7.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.7.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.7.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.7.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.8.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.8.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.8.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.8.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.8.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.8.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.8.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.8.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.8.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.8.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.8.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.8.2.conv.weight"] = original_model[
        "model.diffusion_model.output_blocks.8.2.conv.weight"
    ]
    converted["diffusion"]["denoising_unet.decoders.8.2.conv.bias"] = original_model[
        "model.diffusion_model.output_blocks.8.2.conv.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.9.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.0.out_layers.3.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.residual_connection.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.0.skip_connection.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.9.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.9.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.9.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.9.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.9.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.9.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.9.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.9.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.9.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.9.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.9.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.9.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.10.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.10.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.10.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.10.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.10.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.0.out_layers.3.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.10.0.residual_connection.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.10.0.skip_connection.weight"
    ]
    converted["diffusion"]["denoising_unet.decoders.10.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.10.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.10.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.10.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.10.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.10.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.10.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.10.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.10.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.10.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.10.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.10.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.10.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.10.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.10.1.proj_out.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.feature_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.11.0.in_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.feature_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.0.in_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.feature_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.11.0.in_layers.2.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.feature_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.0.in_layers.2.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.temporal_linear.weight"] = (
        original_model["model.diffusion_model.output_blocks.11.0.emb_layers.1.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.temporal_linear.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.0.emb_layers.1.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.merged_groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.11.0.out_layers.0.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.merged_groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.0.out_layers.0.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.merged_conv.weight"] = (
        original_model["model.diffusion_model.output_blocks.11.0.out_layers.3.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.0.merged_conv.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.0.out_layers.3.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.11.0.residual_connection.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.11.0.skip_connection.weight"
    ]
    converted["diffusion"]["denoising_unet.decoders.11.0.residual_connection.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.0.skip_connection.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.groupnorm.weight"] = (
        original_model["model.diffusion_model.output_blocks.11.1.norm.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.groupnorm.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.1.norm.bias"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.input_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.11.1.proj_in.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.input_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.1.proj_in.bias"]
    )
    converted["diffusion"][
        "denoising_unet.decoders.11.1.self_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.11.1.self_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.11.1.geglu_linear_1.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.geglu_linear_1.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.geglu_linear_2.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.geglu_linear_2.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.bias"
        ]
    )
    converted["diffusion"][
        "denoising_unet.decoders.11.1.cross_attention.query_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_q.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.11.1.cross_attention.key_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.11.1.cross_attention.value_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.11.1.cross_attention.output_proj.weight"
    ] = original_model[
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.weight"
    ]
    converted["diffusion"][
        "denoising_unet.decoders.11.1.cross_attention.output_proj.bias"
    ] = original_model[
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.bias"
    ]
    converted["diffusion"]["denoising_unet.decoders.11.1.self_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.self_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.cross_attn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.cross_attn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.ffn_norm.weight"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.weight"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.ffn_norm.bias"] = (
        original_model[
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.bias"
        ]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.output_projection.weight"] = (
        original_model["model.diffusion_model.output_blocks.11.1.proj_out.weight"]
    )
    converted["diffusion"]["denoising_unet.decoders.11.1.output_projection.bias"] = (
        original_model["model.diffusion_model.output_blocks.11.1.proj_out.bias"]
    )
    converted["diffusion"]["output_layer.groupnorm.weight"] = original_model[
        "model.diffusion_model.out.0.weight"
    ]
    converted["diffusion"]["output_layer.groupnorm.bias"] = original_model[
        "model.diffusion_model.out.0.bias"
    ]
    converted["diffusion"]["output_layer.conv.weight"] = original_model[
        "model.diffusion_model.out.2.weight"
    ]
    converted["diffusion"]["output_layer.conv.bias"] = original_model[
        "model.diffusion_model.out.2.bias"
    ]
    converted["encoder"]["0.weight"] = original_model[
        "first_stage_model.encoder.conv_in.weight"
    ]
    converted["encoder"]["0.bias"] = original_model[
        "first_stage_model.encoder.conv_in.bias"
    ]
    converted["encoder"]["1.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.0.norm1.weight"
    ]
    converted["encoder"]["1.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.0.norm1.bias"
    ]
    converted["encoder"]["1.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.0.conv1.weight"
    ]
    converted["encoder"]["1.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.0.conv1.bias"
    ]
    converted["encoder"]["1.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.0.norm2.weight"
    ]
    converted["encoder"]["1.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.0.norm2.bias"
    ]
    converted["encoder"]["1.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.0.conv2.weight"
    ]
    converted["encoder"]["1.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.0.conv2.bias"
    ]
    converted["encoder"]["2.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.1.norm1.weight"
    ]
    converted["encoder"]["2.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.1.norm1.bias"
    ]
    converted["encoder"]["2.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.1.conv1.weight"
    ]
    converted["encoder"]["2.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.1.conv1.bias"
    ]
    converted["encoder"]["2.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.1.norm2.weight"
    ]
    converted["encoder"]["2.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.1.norm2.bias"
    ]
    converted["encoder"]["2.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.0.block.1.conv2.weight"
    ]
    converted["encoder"]["2.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.0.block.1.conv2.bias"
    ]
    converted["encoder"]["3.weight"] = original_model[
        "first_stage_model.encoder.down.0.downsample.conv.weight"
    ]
    converted["encoder"]["3.bias"] = original_model[
        "first_stage_model.encoder.down.0.downsample.conv.bias"
    ]
    converted["encoder"]["4.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.norm1.weight"
    ]
    converted["encoder"]["4.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.norm1.bias"
    ]
    converted["encoder"]["4.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.conv1.weight"
    ]
    converted["encoder"]["4.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.conv1.bias"
    ]
    converted["encoder"]["4.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.norm2.weight"
    ]
    converted["encoder"]["4.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.norm2.bias"
    ]
    converted["encoder"]["4.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.conv2.weight"
    ]
    converted["encoder"]["4.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.conv2.bias"
    ]
    converted["encoder"]["4.residual_layer.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.0.nin_shortcut.weight"
    ]
    converted["encoder"]["4.residual_layer.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.0.nin_shortcut.bias"
    ]
    converted["encoder"]["5.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.1.norm1.weight"
    ]
    converted["encoder"]["5.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.1.norm1.bias"
    ]
    converted["encoder"]["5.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.1.conv1.weight"
    ]
    converted["encoder"]["5.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.1.conv1.bias"
    ]
    converted["encoder"]["5.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.1.norm2.weight"
    ]
    converted["encoder"]["5.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.1.norm2.bias"
    ]
    converted["encoder"]["5.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.1.block.1.conv2.weight"
    ]
    converted["encoder"]["5.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.1.block.1.conv2.bias"
    ]
    converted["encoder"]["6.weight"] = original_model[
        "first_stage_model.encoder.down.1.downsample.conv.weight"
    ]
    converted["encoder"]["6.bias"] = original_model[
        "first_stage_model.encoder.down.1.downsample.conv.bias"
    ]
    converted["encoder"]["7.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.norm1.weight"
    ]
    converted["encoder"]["7.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.norm1.bias"
    ]
    converted["encoder"]["7.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.conv1.weight"
    ]
    converted["encoder"]["7.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.conv1.bias"
    ]
    converted["encoder"]["7.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.norm2.weight"
    ]
    converted["encoder"]["7.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.norm2.bias"
    ]
    converted["encoder"]["7.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.conv2.weight"
    ]
    converted["encoder"]["7.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.conv2.bias"
    ]
    converted["encoder"]["7.residual_layer.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.0.nin_shortcut.weight"
    ]
    converted["encoder"]["7.residual_layer.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.0.nin_shortcut.bias"
    ]
    converted["encoder"]["8.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.1.norm1.weight"
    ]
    converted["encoder"]["8.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.1.norm1.bias"
    ]
    converted["encoder"]["8.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.1.conv1.weight"
    ]
    converted["encoder"]["8.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.1.conv1.bias"
    ]
    converted["encoder"]["8.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.1.norm2.weight"
    ]
    converted["encoder"]["8.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.1.norm2.bias"
    ]
    converted["encoder"]["8.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.2.block.1.conv2.weight"
    ]
    converted["encoder"]["8.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.2.block.1.conv2.bias"
    ]
    converted["encoder"]["9.weight"] = original_model[
        "first_stage_model.encoder.down.2.downsample.conv.weight"
    ]
    converted["encoder"]["9.bias"] = original_model[
        "first_stage_model.encoder.down.2.downsample.conv.bias"
    ]
    converted["encoder"]["10.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.0.norm1.weight"
    ]
    converted["encoder"]["10.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.0.norm1.bias"
    ]
    converted["encoder"]["10.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.0.conv1.weight"
    ]
    converted["encoder"]["10.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.0.conv1.bias"
    ]
    converted["encoder"]["10.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.0.norm2.weight"
    ]
    converted["encoder"]["10.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.0.norm2.bias"
    ]
    converted["encoder"]["10.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.0.conv2.weight"
    ]
    converted["encoder"]["10.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.0.conv2.bias"
    ]
    converted["encoder"]["11.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.1.norm1.weight"
    ]
    converted["encoder"]["11.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.1.norm1.bias"
    ]
    converted["encoder"]["11.conv_1.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.1.conv1.weight"
    ]
    converted["encoder"]["11.conv_1.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.1.conv1.bias"
    ]
    converted["encoder"]["11.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.1.norm2.weight"
    ]
    converted["encoder"]["11.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.1.norm2.bias"
    ]
    converted["encoder"]["11.conv_2.weight"] = original_model[
        "first_stage_model.encoder.down.3.block.1.conv2.weight"
    ]
    converted["encoder"]["11.conv_2.bias"] = original_model[
        "first_stage_model.encoder.down.3.block.1.conv2.bias"
    ]
    converted["encoder"]["12.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.mid.block_1.norm1.weight"
    ]
    converted["encoder"]["12.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.mid.block_1.norm1.bias"
    ]
    converted["encoder"]["12.conv_1.weight"] = original_model[
        "first_stage_model.encoder.mid.block_1.conv1.weight"
    ]
    converted["encoder"]["12.conv_1.bias"] = original_model[
        "first_stage_model.encoder.mid.block_1.conv1.bias"
    ]
    converted["encoder"]["12.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.mid.block_1.norm2.weight"
    ]
    converted["encoder"]["12.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.mid.block_1.norm2.bias"
    ]
    converted["encoder"]["12.conv_2.weight"] = original_model[
        "first_stage_model.encoder.mid.block_1.conv2.weight"
    ]
    converted["encoder"]["12.conv_2.bias"] = original_model[
        "first_stage_model.encoder.mid.block_1.conv2.bias"
    ]
    converted["encoder"]["13.groupnorm.weight"] = original_model[
        "first_stage_model.encoder.mid.attn_1.norm.weight"
    ]
    converted["encoder"]["13.groupnorm.bias"] = original_model[
        "first_stage_model.encoder.mid.attn_1.norm.bias"
    ]
    converted["encoder"]["13.attention.output_proj.bias"] = original_model[
        "first_stage_model.encoder.mid.attn_1.proj_out.bias"
    ]
    converted["encoder"]["14.groupnorm_1.weight"] = original_model[
        "first_stage_model.encoder.mid.block_2.norm1.weight"
    ]
    converted["encoder"]["14.groupnorm_1.bias"] = original_model[
        "first_stage_model.encoder.mid.block_2.norm1.bias"
    ]
    converted["encoder"]["14.conv_1.weight"] = original_model[
        "first_stage_model.encoder.mid.block_2.conv1.weight"
    ]
    converted["encoder"]["14.conv_1.bias"] = original_model[
        "first_stage_model.encoder.mid.block_2.conv1.bias"
    ]
    converted["encoder"]["14.groupnorm_2.weight"] = original_model[
        "first_stage_model.encoder.mid.block_2.norm2.weight"
    ]
    converted["encoder"]["14.groupnorm_2.bias"] = original_model[
        "first_stage_model.encoder.mid.block_2.norm2.bias"
    ]
    converted["encoder"]["14.conv_2.weight"] = original_model[
        "first_stage_model.encoder.mid.block_2.conv2.weight"
    ]
    converted["encoder"]["14.conv_2.bias"] = original_model[
        "first_stage_model.encoder.mid.block_2.conv2.bias"
    ]
    converted["encoder"]["15.weight"] = original_model[
        "first_stage_model.encoder.norm_out.weight"
    ]
    converted["encoder"]["15.bias"] = original_model[
        "first_stage_model.encoder.norm_out.bias"
    ]
    converted["encoder"]["17.weight"] = original_model[
        "first_stage_model.encoder.conv_out.weight"
    ]
    converted["encoder"]["17.bias"] = original_model[
        "first_stage_model.encoder.conv_out.bias"
    ]
    converted["decoder"]["1.weight"] = original_model[
        "first_stage_model.decoder.conv_in.weight"
    ]
    converted["decoder"]["1.bias"] = original_model[
        "first_stage_model.decoder.conv_in.bias"
    ]
    converted["decoder"]["2.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.mid.block_1.norm1.weight"
    ]
    converted["decoder"]["2.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.mid.block_1.norm1.bias"
    ]
    converted["decoder"]["2.conv_1.weight"] = original_model[
        "first_stage_model.decoder.mid.block_1.conv1.weight"
    ]
    converted["decoder"]["2.conv_1.bias"] = original_model[
        "first_stage_model.decoder.mid.block_1.conv1.bias"
    ]
    converted["decoder"]["2.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.mid.block_1.norm2.weight"
    ]
    converted["decoder"]["2.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.mid.block_1.norm2.bias"
    ]
    converted["decoder"]["2.conv_2.weight"] = original_model[
        "first_stage_model.decoder.mid.block_1.conv2.weight"
    ]
    converted["decoder"]["2.conv_2.bias"] = original_model[
        "first_stage_model.decoder.mid.block_1.conv2.bias"
    ]
    converted["decoder"]["3.groupnorm.weight"] = original_model[
        "first_stage_model.decoder.mid.attn_1.norm.weight"
    ]
    converted["decoder"]["3.groupnorm.bias"] = original_model[
        "first_stage_model.decoder.mid.attn_1.norm.bias"
    ]
    converted["decoder"]["3.attention.output_proj.bias"] = original_model[
        "first_stage_model.decoder.mid.attn_1.proj_out.bias"
    ]
    converted["decoder"]["4.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.mid.block_2.norm1.weight"
    ]
    converted["decoder"]["4.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.mid.block_2.norm1.bias"
    ]
    converted["decoder"]["4.conv_1.weight"] = original_model[
        "first_stage_model.decoder.mid.block_2.conv1.weight"
    ]
    converted["decoder"]["4.conv_1.bias"] = original_model[
        "first_stage_model.decoder.mid.block_2.conv1.bias"
    ]
    converted["decoder"]["4.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.mid.block_2.norm2.weight"
    ]
    converted["decoder"]["4.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.mid.block_2.norm2.bias"
    ]
    converted["decoder"]["4.conv_2.weight"] = original_model[
        "first_stage_model.decoder.mid.block_2.conv2.weight"
    ]
    converted["decoder"]["4.conv_2.bias"] = original_model[
        "first_stage_model.decoder.mid.block_2.conv2.bias"
    ]
    converted["decoder"]["20.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.0.norm1.weight"
    ]
    converted["decoder"]["20.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.0.norm1.bias"
    ]
    converted["decoder"]["20.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.0.conv1.weight"
    ]
    converted["decoder"]["20.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.0.conv1.bias"
    ]
    converted["decoder"]["20.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.0.norm2.weight"
    ]
    converted["decoder"]["20.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.0.norm2.bias"
    ]
    converted["decoder"]["20.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.0.conv2.weight"
    ]
    converted["decoder"]["20.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.0.conv2.bias"
    ]
    converted["decoder"]["20.residual_layer.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.0.nin_shortcut.weight"
    ]
    converted["decoder"]["20.residual_layer.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.0.nin_shortcut.bias"
    ]
    converted["decoder"]["21.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.1.norm1.weight"
    ]
    converted["decoder"]["21.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.1.norm1.bias"
    ]
    converted["decoder"]["21.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.1.conv1.weight"
    ]
    converted["decoder"]["21.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.1.conv1.bias"
    ]
    converted["decoder"]["21.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.1.norm2.weight"
    ]
    converted["decoder"]["21.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.1.norm2.bias"
    ]
    converted["decoder"]["21.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.1.conv2.weight"
    ]
    converted["decoder"]["21.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.1.conv2.bias"
    ]
    converted["decoder"]["22.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.2.norm1.weight"
    ]
    converted["decoder"]["22.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.2.norm1.bias"
    ]
    converted["decoder"]["22.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.2.conv1.weight"
    ]
    converted["decoder"]["22.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.2.conv1.bias"
    ]
    converted["decoder"]["22.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.2.norm2.weight"
    ]
    converted["decoder"]["22.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.2.norm2.bias"
    ]
    converted["decoder"]["22.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.0.block.2.conv2.weight"
    ]
    converted["decoder"]["22.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.0.block.2.conv2.bias"
    ]
    converted["decoder"]["15.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.0.norm1.weight"
    ]
    converted["decoder"]["15.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.0.norm1.bias"
    ]
    converted["decoder"]["15.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.0.conv1.weight"
    ]
    converted["decoder"]["15.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.0.conv1.bias"
    ]
    converted["decoder"]["15.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.0.norm2.weight"
    ]
    converted["decoder"]["15.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.0.norm2.bias"
    ]
    converted["decoder"]["15.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.0.conv2.weight"
    ]
    converted["decoder"]["15.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.0.conv2.bias"
    ]
    converted["decoder"]["15.residual_layer.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.0.nin_shortcut.weight"
    ]
    converted["decoder"]["15.residual_layer.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.0.nin_shortcut.bias"
    ]
    converted["decoder"]["16.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.1.norm1.weight"
    ]
    converted["decoder"]["16.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.1.norm1.bias"
    ]
    converted["decoder"]["16.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.1.conv1.weight"
    ]
    converted["decoder"]["16.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.1.conv1.bias"
    ]
    converted["decoder"]["16.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.1.norm2.weight"
    ]
    converted["decoder"]["16.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.1.norm2.bias"
    ]
    converted["decoder"]["16.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.1.conv2.weight"
    ]
    converted["decoder"]["16.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.1.conv2.bias"
    ]
    converted["decoder"]["17.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.2.norm1.weight"
    ]
    converted["decoder"]["17.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.2.norm1.bias"
    ]
    converted["decoder"]["17.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.2.conv1.weight"
    ]
    converted["decoder"]["17.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.2.conv1.bias"
    ]
    converted["decoder"]["17.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.2.norm2.weight"
    ]
    converted["decoder"]["17.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.2.norm2.bias"
    ]
    converted["decoder"]["17.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.1.block.2.conv2.weight"
    ]
    converted["decoder"]["17.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.1.block.2.conv2.bias"
    ]
    converted["decoder"]["19.weight"] = original_model[
        "first_stage_model.decoder.up.1.upsample.conv.weight"
    ]
    converted["decoder"]["19.bias"] = original_model[
        "first_stage_model.decoder.up.1.upsample.conv.bias"
    ]
    converted["decoder"]["10.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.0.norm1.weight"
    ]
    converted["decoder"]["10.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.0.norm1.bias"
    ]
    converted["decoder"]["10.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.0.conv1.weight"
    ]
    converted["decoder"]["10.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.0.conv1.bias"
    ]
    converted["decoder"]["10.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.0.norm2.weight"
    ]
    converted["decoder"]["10.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.0.norm2.bias"
    ]
    converted["decoder"]["10.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.0.conv2.weight"
    ]
    converted["decoder"]["10.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.0.conv2.bias"
    ]
    converted["decoder"]["11.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.1.norm1.weight"
    ]
    converted["decoder"]["11.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.1.norm1.bias"
    ]
    converted["decoder"]["11.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.1.conv1.weight"
    ]
    converted["decoder"]["11.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.1.conv1.bias"
    ]
    converted["decoder"]["11.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.1.norm2.weight"
    ]
    converted["decoder"]["11.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.1.norm2.bias"
    ]
    converted["decoder"]["11.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.1.conv2.weight"
    ]
    converted["decoder"]["11.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.1.conv2.bias"
    ]
    converted["decoder"]["12.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.2.norm1.weight"
    ]
    converted["decoder"]["12.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.2.norm1.bias"
    ]
    converted["decoder"]["12.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.2.conv1.weight"
    ]
    converted["decoder"]["12.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.2.conv1.bias"
    ]
    converted["decoder"]["12.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.2.norm2.weight"
    ]
    converted["decoder"]["12.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.2.norm2.bias"
    ]
    converted["decoder"]["12.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.2.block.2.conv2.weight"
    ]
    converted["decoder"]["12.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.2.block.2.conv2.bias"
    ]
    converted["decoder"]["14.weight"] = original_model[
        "first_stage_model.decoder.up.2.upsample.conv.weight"
    ]
    converted["decoder"]["14.bias"] = original_model[
        "first_stage_model.decoder.up.2.upsample.conv.bias"
    ]
    converted["decoder"]["5.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.0.norm1.weight"
    ]
    converted["decoder"]["5.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.0.norm1.bias"
    ]
    converted["decoder"]["5.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.0.conv1.weight"
    ]
    converted["decoder"]["5.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.0.conv1.bias"
    ]
    converted["decoder"]["5.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.0.norm2.weight"
    ]
    converted["decoder"]["5.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.0.norm2.bias"
    ]
    converted["decoder"]["5.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.0.conv2.weight"
    ]
    converted["decoder"]["5.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.0.conv2.bias"
    ]
    converted["decoder"]["6.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.1.norm1.weight"
    ]
    converted["decoder"]["6.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.1.norm1.bias"
    ]
    converted["decoder"]["6.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.1.conv1.weight"
    ]
    converted["decoder"]["6.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.1.conv1.bias"
    ]
    converted["decoder"]["6.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.1.norm2.weight"
    ]
    converted["decoder"]["6.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.1.norm2.bias"
    ]
    converted["decoder"]["6.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.1.conv2.weight"
    ]
    converted["decoder"]["6.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.1.conv2.bias"
    ]
    converted["decoder"]["7.groupnorm_1.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.2.norm1.weight"
    ]
    converted["decoder"]["7.groupnorm_1.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.2.norm1.bias"
    ]
    converted["decoder"]["7.conv_1.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.2.conv1.weight"
    ]
    converted["decoder"]["7.conv_1.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.2.conv1.bias"
    ]
    converted["decoder"]["7.groupnorm_2.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.2.norm2.weight"
    ]
    converted["decoder"]["7.groupnorm_2.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.2.norm2.bias"
    ]
    converted["decoder"]["7.conv_2.weight"] = original_model[
        "first_stage_model.decoder.up.3.block.2.conv2.weight"
    ]
    converted["decoder"]["7.conv_2.bias"] = original_model[
        "first_stage_model.decoder.up.3.block.2.conv2.bias"
    ]
    converted["decoder"]["9.weight"] = original_model[
        "first_stage_model.decoder.up.3.upsample.conv.weight"
    ]
    converted["decoder"]["9.bias"] = original_model[
        "first_stage_model.decoder.up.3.upsample.conv.bias"
    ]
    converted["decoder"]["23.weight"] = original_model[
        "first_stage_model.decoder.norm_out.weight"
    ]
    converted["decoder"]["23.bias"] = original_model[
        "first_stage_model.decoder.norm_out.bias"
    ]
    converted["decoder"]["25.weight"] = original_model[
        "first_stage_model.decoder.conv_out.weight"
    ]
    converted["decoder"]["25.bias"] = original_model[
        "first_stage_model.decoder.conv_out.bias"
    ]
    converted["encoder"]["18.weight"] = original_model[
        "first_stage_model.quant_conv.weight"
    ]
    converted["encoder"]["18.bias"] = original_model[
        "first_stage_model.quant_conv.bias"
    ]
    converted["decoder"]["0.weight"] = original_model[
        "first_stage_model.post_quant_conv.weight"
    ]
    converted["decoder"]["0.bias"] = original_model[
        "first_stage_model.post_quant_conv.bias"
    ]
    converted["clip"]["embedding.token_embedding.weight"] = original_model[
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
    ]
    converted["clip"]["embedding.position_embedding"] = original_model[
        "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"
    ]
    converted["clip"]["layers.0.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.0.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.0.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight"
    ]
    converted["clip"]["layers.0.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias"
    ]
    converted["clip"]["layers.0.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight"
    ]
    converted["clip"]["layers.0.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.bias"
    ]
    converted["clip"]["layers.0.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.weight"
    ]
    converted["clip"]["layers.0.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.bias"
    ]
    converted["clip"]["layers.0.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.weight"
    ]
    converted["clip"]["layers.0.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.bias"
    ]
    converted["clip"]["layers.1.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.1.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.1.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.weight"
    ]
    converted["clip"]["layers.1.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.bias"
    ]
    converted["clip"]["layers.1.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.weight"
    ]
    converted["clip"]["layers.1.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.bias"
    ]
    converted["clip"]["layers.1.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.weight"
    ]
    converted["clip"]["layers.1.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.bias"
    ]
    converted["clip"]["layers.1.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.weight"
    ]
    converted["clip"]["layers.1.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.bias"
    ]
    converted["clip"]["layers.2.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.2.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.2.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.weight"
    ]
    converted["clip"]["layers.2.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.bias"
    ]
    converted["clip"]["layers.2.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.weight"
    ]
    converted["clip"]["layers.2.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.bias"
    ]
    converted["clip"]["layers.2.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.weight"
    ]
    converted["clip"]["layers.2.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.bias"
    ]
    converted["clip"]["layers.2.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.weight"
    ]
    converted["clip"]["layers.2.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.bias"
    ]
    converted["clip"]["layers.3.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.3.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.3.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.weight"
    ]
    converted["clip"]["layers.3.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.bias"
    ]
    converted["clip"]["layers.3.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.weight"
    ]
    converted["clip"]["layers.3.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.bias"
    ]
    converted["clip"]["layers.3.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.weight"
    ]
    converted["clip"]["layers.3.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.bias"
    ]
    converted["clip"]["layers.3.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.weight"
    ]
    converted["clip"]["layers.3.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.bias"
    ]
    converted["clip"]["layers.4.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.4.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.4.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.weight"
    ]
    converted["clip"]["layers.4.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.bias"
    ]
    converted["clip"]["layers.4.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.weight"
    ]
    converted["clip"]["layers.4.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.bias"
    ]
    converted["clip"]["layers.4.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.weight"
    ]
    converted["clip"]["layers.4.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.bias"
    ]
    converted["clip"]["layers.4.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.weight"
    ]
    converted["clip"]["layers.4.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.bias"
    ]
    converted["clip"]["layers.5.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.5.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.5.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.weight"
    ]
    converted["clip"]["layers.5.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.bias"
    ]
    converted["clip"]["layers.5.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.weight"
    ]
    converted["clip"]["layers.5.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.bias"
    ]
    converted["clip"]["layers.5.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.weight"
    ]
    converted["clip"]["layers.5.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.bias"
    ]
    converted["clip"]["layers.5.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.weight"
    ]
    converted["clip"]["layers.5.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.bias"
    ]
    converted["clip"]["layers.6.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.6.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.6.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.weight"
    ]
    converted["clip"]["layers.6.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.bias"
    ]
    converted["clip"]["layers.6.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.weight"
    ]
    converted["clip"]["layers.6.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.bias"
    ]
    converted["clip"]["layers.6.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.weight"
    ]
    converted["clip"]["layers.6.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.bias"
    ]
    converted["clip"]["layers.6.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.weight"
    ]
    converted["clip"]["layers.6.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.bias"
    ]
    converted["clip"]["layers.7.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.7.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.7.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.weight"
    ]
    converted["clip"]["layers.7.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.bias"
    ]
    converted["clip"]["layers.7.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.weight"
    ]
    converted["clip"]["layers.7.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.bias"
    ]
    converted["clip"]["layers.7.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.weight"
    ]
    converted["clip"]["layers.7.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.bias"
    ]
    converted["clip"]["layers.7.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.weight"
    ]
    converted["clip"]["layers.7.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.bias"
    ]
    converted["clip"]["layers.8.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.8.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.8.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.weight"
    ]
    converted["clip"]["layers.8.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.bias"
    ]
    converted["clip"]["layers.8.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.weight"
    ]
    converted["clip"]["layers.8.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.bias"
    ]
    converted["clip"]["layers.8.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.weight"
    ]
    converted["clip"]["layers.8.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.bias"
    ]
    converted["clip"]["layers.8.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.weight"
    ]
    converted["clip"]["layers.8.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.bias"
    ]
    converted["clip"]["layers.9.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.9.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.9.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.weight"
    ]
    converted["clip"]["layers.9.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.bias"
    ]
    converted["clip"]["layers.9.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.weight"
    ]
    converted["clip"]["layers.9.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.bias"
    ]
    converted["clip"]["layers.9.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.weight"
    ]
    converted["clip"]["layers.9.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.bias"
    ]
    converted["clip"]["layers.9.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.weight"
    ]
    converted["clip"]["layers.9.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.bias"
    ]
    converted["clip"]["layers.10.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.10.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.10.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.weight"
    ]
    converted["clip"]["layers.10.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.bias"
    ]
    converted["clip"]["layers.10.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.weight"
    ]
    converted["clip"]["layers.10.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.bias"
    ]
    converted["clip"]["layers.10.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.weight"
    ]
    converted["clip"]["layers.10.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.bias"
    ]
    converted["clip"]["layers.10.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.weight"
    ]
    converted["clip"]["layers.10.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.bias"
    ]
    converted["clip"]["layers.11.attention.output_proj.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.weight"
    ]
    converted["clip"]["layers.11.attention.output_proj.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.bias"
    ]
    converted["clip"]["layers.11.layernorm_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.weight"
    ]
    converted["clip"]["layers.11.layernorm_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.bias"
    ]
    converted["clip"]["layers.11.linear_1.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.weight"
    ]
    converted["clip"]["layers.11.linear_1.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.bias"
    ]
    converted["clip"]["layers.11.linear_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.weight"
    ]
    converted["clip"]["layers.11.linear_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.bias"
    ]
    converted["clip"]["layers.11.layernorm_2.weight"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.weight"
    ]
    converted["clip"]["layers.11.layernorm_2.bias"] = original_model[
        "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.bias"
    ]
    converted["clip"]["layernorm.weight"] = original_model[
        "cond_stage_model.transformer.text_model.final_layer_norm.weight"
    ]
    converted["clip"]["layernorm.bias"] = original_model[
        "cond_stage_model.transformer.text_model.final_layer_norm.bias"
    ]
    converted["diffusion"][
        "denoising_unet.encoders.1.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.encoders.2.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.encoders.4.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.encoders.5.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.encoders.7.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.encoders.8.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.bottleneck.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.3.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.4.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.5.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.6.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.7.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.8.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.9.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.10.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["diffusion"][
        "denoising_unet.decoders.11.1.self_attention.input_proj.weight"
    ] = torch.cat(
        (
            original_model[
                "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_q.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.weight"
            ],
            original_model[
                "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_v.weight"
            ],
        ),
        0,
    )
    converted["encoder"]["13.attention.input_proj.weight"] = torch.cat(
        (
            original_model["first_stage_model.encoder.mid.attn_1.q.weight"],
            original_model["first_stage_model.encoder.mid.attn_1.k.weight"],
            original_model["first_stage_model.encoder.mid.attn_1.v.weight"],
        ),
        0,
    ).reshape((1536, 512))
    converted["encoder"]["13.attention.input_proj.bias"] = torch.cat(
        (
            original_model["first_stage_model.encoder.mid.attn_1.q.bias"],
            original_model["first_stage_model.encoder.mid.attn_1.k.bias"],
            original_model["first_stage_model.encoder.mid.attn_1.v.bias"],
        ),
        0,
    )
    converted["encoder"]["13.attention.output_proj.weight"] = original_model[
        "first_stage_model.encoder.mid.attn_1.proj_out.weight"
    ].reshape((512, 512))
    converted["decoder"]["3.attention.input_proj.weight"] = torch.cat(
        (
            original_model["first_stage_model.decoder.mid.attn_1.q.weight"],
            original_model["first_stage_model.decoder.mid.attn_1.k.weight"],
            original_model["first_stage_model.decoder.mid.attn_1.v.weight"],
        ),
        0,
    ).reshape((1536, 512))
    converted["decoder"]["3.attention.input_proj.bias"] = torch.cat(
        (
            original_model["first_stage_model.decoder.mid.attn_1.q.bias"],
            original_model["first_stage_model.decoder.mid.attn_1.k.bias"],
            original_model["first_stage_model.decoder.mid.attn_1.v.bias"],
        ),
        0,
    )
    converted["decoder"]["3.attention.output_proj.weight"] = original_model[
        "first_stage_model.decoder.mid.attn_1.proj_out.weight"
    ].reshape((512, 512))
    converted["clip"]["layers.0.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.0.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.1.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.1.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.2.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.2.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.3.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.3.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.4.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.4.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.5.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.5.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.6.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.6.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.7.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.7.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.8.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.8.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.9.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.9.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.10.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.10.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.bias"
            ],
        ),
        0,
    )
    converted["clip"]["layers.11.attention.input_proj.weight"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.weight"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.weight"
            ],
        ),
        0,
    )
    converted["clip"]["layers.11.attention.input_proj.bias"] = torch.cat(
        (
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.bias"
            ],
            original_model[
                "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.bias"
            ],
        ),
        0,
    )

    return converted
