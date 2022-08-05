import torch


def flatten_activations(activations):
    if isinstance(activations, torch.Tensor):
        return activations.view(-1, activations.shape[-1])
    elif isinstance(activations, list):
        return torch.cat([flatten_activations(act) for act in activations])
    elif isinstance(activations, dict):
        return flatten_activations(list(activations.values()))


def make_projection(dir):
    norm_dir = dir / torch.linalg.norm(dir)

    def project(m, i, o):
        hidden_states, rest = o
        hidden_states -= torch.einsum(
            "b n h, h, k -> b n k", hidden_states, norm_dir, norm_dir
        )
        return hidden_states, rest

    return project


def orthonormalize(vs):
    for i, v in enumerate(vs):
        for j in range(i):
            v -= (vs[j] @ v) * vs[j]
        vs[i] = v / torch.linalg.norm(v)
    return vs


def proj_on(x, vs, device="cpu"):
    r = torch.zeros(vs.shape[1]).to(device)
    for v in vs:
        r += (v @ x) * v
    return r


def make_projections(dirs):
    def project(m, i, o):
        hidden_states, rest = o
        for norm_dir in dirs:
            hidden_states -= torch.einsum(
                "b n h, h, k -> b n k", hidden_states, norm_dir, norm_dir
            )
        return hidden_states, rest

    return project
