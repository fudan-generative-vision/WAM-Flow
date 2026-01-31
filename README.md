<h1 align='center'>WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving</h1>
<div align='center'>
    <a href='https://github.com/YoucanBaby' target='_blank'>Yifang Xu</a><sup>1*</sup>&emsp;
    <a href='https://cuijh26.github.io/' target='_blank'>Jiahao Cui</a><sup>1*</sup>&emsp;
    <a href='https://github.com/fudan-generative-vision/WAM-Flow' target='_blank'>Feipeng Cai</a><sup>2*</sup>&emsp;
    <a href='https://github.com/SSSSSSuger' target='_blank'>Zhihao Zhu</a><sup>1</sup>&emsp;
    <a href='https://github.com/NinoNeumann' target='_blank'>Hanlin Shang</a><sup>1</sup>&emsp;
    <a href='https://github.com/isan089' target='_blank'>Shan Luan</a><sup>1</sup>&emsp;
</div>
<div align='center'>
    <a href='https://github.com/xumingw' target='_blank'>Mingwang Xu</a><sup>1</sup>&emsp;
    <a href='https://github.com/fudan-generative-vision/WAM-Flow' target='_blank'>Neng Zhang</a><sup>2</sup>&emsp;
    <a href='https://github.com/fudan-generative-vision/WAM-Flow' target='_blank'>Yaoyi Li</a><sup>2</sup>&emsp;
    <a href='https://github.com/fudan-generative-vision/WAM-Flow‘ target='_blank'>Jia Cai</a><sup>2</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>1</sup>&emsp;
</div>

<div align='center'>
    <sup>1</sup>Fudan University&emsp; <sup>2</sup>Yinwang Intelligent Technology Co., Ltd&emsp;
</div>

<br>
<div align='center'>
    <a href='https://github.com/fudan-generative-vision/WAM-Flow'><img src='https://img.shields.io/github/stars/fudan-generative-vision/WAM-Flow?style=social'></a>
    <a href='https://arxiv.org/abs/2512.06112'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/fudan-generative-ai/WAM-Flow'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
</div>
<br>



## 📰 News
- **`2026/02/01`**: 🎉🎉🎉 Release the pretrained models on [Huggingface](https://huggingface.co/fudan-generative-ai/WAM-Flow).
- **`2025/12/06`**: 🎉🎉🎉 Paper submitted on [Arxiv](https://arxiv.org/pdf/2512.06112).



## 📅️ Roadmap

| Status | Milestone                                                                                             |    ETA     |
| :----: | :----------------------------------------------------------------------------------------------------: | :--------: |
|   ✅   | **[Release the SFT and inference code](https://github.com/fudan-generative-vision/WAM-Flow)**   | 2025.12.19 |
|   ✅   | **[Pretrained models on Huggingface](https://huggingface.co/fudan-generative-ai/WAM-Flow)**    | 2026.02.01        |
|   🚀   | **[Release the evaluation code](https://huggingface.co/fudan-generative-ai/WAM-Flow)**    | TBD |
|   🚀   | **[Release the RL code](https://github.com/fudan-generative-vision/WAM-Flow)**   | TBD |
|   🚀   | **[Release the pre-processed training data](#training)**                                       | TBD        |


## 📸 Showcase
![teaser](assets/Figure_1.png)

## 🏆 Qualitative Results on NAVSIM
### NAVSIM-v1 benchmark results
<div style="text-align: center;">
  <img src="assets/navsim-v1.png" alt="navsim-v1" width="70%" />
</div>

### NAVSIM-v2 benchmark results
<div style="text-align: center;">
<img src="assets/navsim-v2.png" alt="navsim-v2" width="70%" />
</div>



## 🔧️ Framework
![framework](assets/Figure_2.png)
Our method takes as input a front-view image, a natural-language navigation command with a system prompt, and the ego-vehicle states, and outputs an 8-waypoint future trajectory spanning 4 seconds through parallel denoising. The model is first trained via supervised fine-tuning to learn accurate trajectory prediction. We then apply simulatorguided GRPO to further optimize closed-loop behavior. The GRPO reward function integrates safety constraints (collision avoidance, drivable-area compliance) with performance objectives (ego-progress, time-to-collision, comfort).



## Quick Start

### Installation

Clone the repo:

```sh
git clone https://github.com/fudan-generative-vision/WAM-Flow.git
cd WAM-Flow
```

Install dependencies:

```sh
conda create --name wam-flow python=3.10
conda activate wam-flow
pip install -r requirements.txt
```


### Model Download

Download models using huggingface-cli:

```sh
pip install "huggingface_hub[cli]"
huggingface-cli download fudan-generative-ai/WAM-Flow --local-dir ./pretrained_model/wam-flow
huggingface-cli download LucasJinWang/FUDOKI --local-dir ./pretrained_model/fudoki
```



### Inference

```sh
sh script/infer.sh
```


### Training

```bash
sh script/sft_debug.sh
```



## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

```
@article{xu2025wam,
  title={WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving},
  author={Xu, Yifang and Cui, Jiahao and Cai, Feipeng and Zhu, Zhihao and Shang, Hanlin and Luan, Shan and Xu, Mingwang and Zhang, Neng and Li, Yaoyi and Cai, Jia and others},
  journal={arXiv preprint arXiv:2512.06112},
  year={2025}
}
```



## ⚠️ Social Risks and Mitigations

The integration of Vision-Language-Action models into autonomous driving introduces ethical challenges, particularly regarding the opacity of neural decision-making and its impact on road safety. To mitigate these risks, it is imperative to implement explainable AI frameworks and robust safe protocols that ensure predictable vehicle behavior in long-tailed scenarios. Furthermore, addressing concerns over data privacy and public surveillance requires transparent data governance and rigorous de-identification practices. By prioritizing safety-critical alignment and ethical compliance, this research promotes the responsible development and deployment of VLA-based autonomous systems.



## 🤗 Acknowledgements
We gratefully acknowledge the contributors to the [Recogdrive](https://github.com/xiaomi-research/recogdrive), [Janus](https://github.com/deepseek-ai/Janus), [FUDOKI](https://github.com/fudoki-hku/FUDOKI) and [flow_matching](https://github.com/facebookresearch/flow_matching) repositories, whose commitment to open source has provided us with their excellent codebases and pretrained models.
