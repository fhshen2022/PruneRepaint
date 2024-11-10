<div align="center">
<h1>PruneRepaint (NeurIPS2024)</h1>
<h3>Prune and Repaint: Content-Aware Image Retargeting for any Ratio</h3>
Paper: ([arXiv:2410.22865](https://arxiv.org/abs/2410.22865))
</div>

## Introduction
PruneRepaint is an image retargeting method that introduces high-level semantic information and  low-level structural prior to the retargeting process. This repository contains the code of PruneRepaint for retargeting an image to any aspect ratio. For more information, please refer to our paper.

<p align="center">
  <img src="./assets/PruneRepaint.jpg" width="800" />
</p>



## Quick Start

To retarget images, use the following commands:

```bash
python PruneRepaint.py  --target_ratio "4/3" --save_image_path "/path/to/save/" --input_image_path "/path/to/original/images/"
```

## Results on [RetargetMe](https://people.csail.mit.edu/mrub/retargetme/) Dataset

### Quantitative Results on Saliency Discard Ratio (SDR)
|Aspect Ratio | 16/9  | 4/3 | 1/1 | 9/16 |
| :--: | :--: | :--: | :--: | :--: |
| Scale | 0.571 | 0.446 | 0.307 | 0.222 |
| Crop | 0.386 | 0.259 | 0.129 | 0.094 |
| Seam-carving | 0.490 | 0.367 | 0.242 | 0.161 |
| InGAN | 0.569 | 0.442 | 0.263 | 0.222 |
| FR | 0.524 | 0.423 | 0.294 | 0.214 |
| Ours | **0.151** | **0.074** | **0.031** | **0.006** |

### User Study on Aspect Ratio 16:9
| Settings | Content Completeness Score | Deformation Score | Local Smoothness Score | Aesthetic Score | Average Score |
| :--: |:--: | :--: | :--: | :--: | :--: |
| Scale |**2.875** | 0.975 | 1.878 | 1.153 | 1.720 | 
| Crop | 1.295 | **2.905** | **2.926** | 2.355 | 2.370 |
| Seam-carving | 2.829 | 0.973 | 1.000 | 1.038 | 1.461 |
| InGAN | 1.662 | 0.975 | 1.007 | 0.866 | 1.126 |
| FR | 1.327 | 1.812 | 1.702 | 1.535 |  1.594 |
| Ours | 2.345 | 2.757 | 2.689 | **2.538** | **2.582** |


## Citation

If PruneRepaint is helpful for your research, please cite the following paper:
```
@inproceedings{shen2024prunerepaint,
	title={Prune and Repaint: Content-Aware Image Retargeting for any Ratio}, 
	author={Feihong Shen and Chao Li and Yifeng Geng and Yongjian Deng and Hao Chen},
	journal={Advances in Neural Information Processing Systems},
	year={2024},
}
```



## Acknowledgment

This project is based on Stable Diffusion([paper](https://arxiv.org/abs/2112.10752), [code](https://github.com/CompVis/latent-diffusion)), IP-Adapter ([paper](https://arxiv.org/abs/2308.06721), [code](https://github.com/tencent-ailab/IP-Adapter)), ControlNet ([paper](https://arxiv.org/abs/2302.05543), [code](https://github.com/lllyasviel/ControlNet)), VST ([paper](https://arxiv.org/abs/2104.12099), [code](https://github.com/nnizhang/VST)), SeamCarving ([paper](https://dl.acm.org/doi/10.1145/1276377.1276390)) and RetargetMe ([paper](https://people.csail.mit.edu/mrub/papers/retBenchmark.pdf)), thanks for their excellent works.

