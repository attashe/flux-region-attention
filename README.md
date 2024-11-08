# flux-region-attention

This repository contains the code for the novel approach to region attention masking in ViT models. The code is based on the Omost repository and UI code is based on the GLIGEN GUI repository.

## Explanation

For encoding multiple regions text prompts we concatenate the text embeddings to one text tensor. Then we create a mask tensor for separate regions in self-attention layers.

## Examples

1. Interface view
![Example 1](./images/example_1.png)

2. Intersecting regions
![Example 2](./images/example_2.png)

3. Result example
![Example 3](./images/example_3.png)

## Aknowledgements

This repository is base on next repositories:

@Misc{omost,
  author = {Omost Team},
  title  = {Omost GitHub Page},
  year   = {2024},
}

[Gligen-GUI](https://github.com/mut-ex/gligen-gui)

[black-forest-labs](https://github.com/black-forest-labs/flux)

[lucidrains attention implementation](https://github.com/lucidrains/memory-efficient-attention-pytorch)