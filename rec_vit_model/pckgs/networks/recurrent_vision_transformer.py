# taken and modified from https://github.com/huggingface/pytorch-image-models

import timm
import torch

from timm.models._manipulate import checkpoint_seq
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from timm.models.vision_transformer import checkpoint_filter_fn
from functools import partial


class RecurrentVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Recurrent Vision Transformer
    """

    def _pos_embed(self, x, cls_tok):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((cls_tok, x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, cls_tok):
        x = self.patch_embed(x)
        x = self._pos_embed(x, cls_tok)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, cls_tok):
        x = self.forward_features(x, cls_tok)
        new_cls_token = x[:, 0]
        return self.forward_head(x), new_cls_token[:, None, :]


def _create_recurrent_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    return build_model_with_cfg(
        RecurrentVisionTransformer, variant, pretrained,
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )


@register_model
def recurrent_vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_recurrent_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
