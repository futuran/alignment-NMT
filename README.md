# alignment-NMT

Transformer-based MT model with build-in word alignment.

## Environments
- (RTX A6000 GPU)
- CUDA 11.1
- torch 1.9.1
- torchtext 0.10.1

This code is based on [OpenNMT-py v2.1.2](https://github.com/OpenNMT/OpenNMT-py/releases/tag/2.1.2). 
Although original script requires torchtext of v0.8.1 or earlier,
this script optimized for v0.9.0 or later to work with Nvidia Ampere series GPU.

## DATASET format
This model is trained using three types of teacher data as follows.
- Input sentences(Source language sentences) + Similar sentences(Target language sentences)
- Reference sentences(Target language sentences)
- Adjacency Matrix