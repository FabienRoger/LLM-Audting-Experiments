import torch


def print_logit_lense(model, tokenizer, dir, topk=5):
    dir = dir.to(model.device)
    for t in torch.topk(model.lm_head(dir), topk).indices:
        print(tokenizer.decode(t), end=" ")
    print("vs ", end="")
    for t in torch.topk(-model.lm_head(dir), topk).indices:
        print(tokenizer.decode(t), end=" ")
    print()
