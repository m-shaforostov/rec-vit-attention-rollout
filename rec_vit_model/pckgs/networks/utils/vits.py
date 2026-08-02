import collections
from functools import partial
import torchvision


def get_rvit_activations(model, data, batch_size, repeats=1, layer=0, dim=72, num_heads=6, blur='no_blur'):
    """
    Inspiration from here:
    https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
    """
    model.eval()
    cls_token = model.cls_token.expand(data.shape[0], -1, -1)
    result = []
    outs = []
    tokens = [cls_token]

    def save_activation(name_l, mod, inp, out_l):
        activations[name_l] = out_l

    for name, m in model.named_modules():
        m.register_forward_hook(partial(save_activation, name))

    # ----------------------------------------------
    if blur != 'no_blur':
        blr = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=4)
        blurred_data = [data]
        for i in range(repeats - 1):
            blurred_data.append(blr(blurred_data[-1]))

        if blur == 'inv_blur':
            blurred_data = blurred_data[::-1]
    # ----------------------------------------------

    for k in range(repeats):
        activations = collections.defaultdict(list)

        out, cls_token = model(blurred_data[k] if blur != 'no_blur' else data, cls_token)
        outs.append(out.data.max(1, keepdim=True)[1])
        activations = {itm: outputs for itm, outputs in activations.items()}
        tokens.append(cls_token)

        if isinstance(layer, int):
            q, k = activations[f'blocks.{layer}.attn.q_norm'][0], activations[f'blocks.{layer}.attn.k_norm'][0]
            scale = (dim // num_heads) ** -0.5
            q = q * scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            result.append(attn)
        elif isinstance(layer, list):
            for la in layer:
                q, k = activations[f'blocks.{la}.attn.q_norm'][0], activations[f'blocks.{la}.attn.k_norm'][0]
                scale = (dim // num_heads) ** -0.5
                q = q * scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                result.append(attn)

    return result, tokens, outs
