# Free-T2M: Frequency-Enhanced Text-to-Motion Diffusion Model 🚀

Free-T2M is a state-of-the-art framework for **text-to-motion generation**, introducing **frequency-domain analysis** and **stage-specific consistency losses**. By focusing on low-frequency semantic alignment and fine-grained motion details, Free-T2M delivers unmatched performance across benchmarks. 🌟

![Framework Diagram](Visualization/Figure/Framework.png)

---

## Key Features 🛠️

- **Frequency-Domain Analysis**: Utilizes low-frequency components for enhanced semantic alignment, ensuring smoother and more natural motions. 📊
- **Consistency Losses**: Combines **low-frequency consistency** and **semantic consistency** losses to minimize artifacts and improve realism. ⚖️
- **State-of-the-Art Results**: Demonstrates superior performance on the **HumanML3D** and **KIT-ML** benchmarks. 🏆

---

## Why Choose Free-T2M? 🤔

Free-T2M integrates advanced frequency-domain techniques and robust consistency mechanisms, setting a new standard in text-to-motion generation. Its versatility spans applications in **animation**, **robotics**, and **virtual reality**, offering unmatched precision and quality. 🎯

---

## Performance Highlights 📈

- **FID Reduction**: Improved FID on the MDM baseline from **0.544** to **0.256**.
- **SOTA on StableMoFusion**: Achieved FID reduction from **0.189** to **0.051**.
- **Human Evaluations**: Significant improvements in user preference and subjective quality assessments.

![Experiment Results](Visualization/Figure/Experiment.png)

---

## Denoising Process 🎥

Free-T2M refines motion generation through a staged denoising process. The visualizations below illustrate the transition from noise to high-quality motion:

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
  <img src="Visualization/Noise/21_0.gif" width="200" style="margin: 10px;">
  <img src="Visualization/Noise/21_10.gif" width="200" style="margin: 10px;">
  <img src="Visualization/Noise/21_20.gif" width="200" style="margin: 10px;">
  <img src="Visualization/Noise/21_30.gif" width="200" style="margin: 10px;">
  <img src="Visualization/Noise/21_40.gif" width="200" style="margin: 10px;">
  <img src="Visualization/Noise/21_50.gif" width="200" style="margin: 10px;">
</div>

---

## Visual Comparisons 🎬

### MDM Baseline vs. Free-T2M

The following GIFs compare Free-T2M with the MDM baseline. Free-T2M produces more realistic and semantically aligned motions:

<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <div>
    <p align="center"><b>Ours</b></p>
    <img src="More_result/MDM/Ours/00.gif" width="200" style="margin: 10px;">
    <img src="More_result/MDM/Ours/01.gif" width="200" style="margin: 10px;">
    <img src="More_result/MDM/Ours/02.gif" width="200" style="margin: 10px;">
  </div>
  <div>
    <p align="center"><b>Baseline</b></p>
    <img src="More_result/MDM/base/00.gif" width="200" style="margin: 10px;">
    <img src="More_result/MDM/base/01.gif" width="200" style="margin: 10px;">
    <img src="More_result/MDM/base/02.gif" width="200" style="margin: 10px;">
  </div>
</div>

---

### StableMoFusion Baseline vs. Free-T2M

Free-T2M outperforms StableMoFusion in generating more realistic and semantically aligned motions:

<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <div>
    <p align="center"><b>Ours</b></p>
    <img src="More_result/StableMofusion/ours/04.gif" width="200" style="margin: 10px;">
    <img src="More_result/StableMofusion/ours/03.gif" width="200" style="margin: 10px;">
    <img src="More_result/StableMofusion/ours/02.gif" width="200" style="margin: 10px;">
  </div>
  <div>
    <p align="center"><b>Baseline</b></p>
    <img src="More_result/StableMofusion/base/04.gif" width="200" style="margin: 10px;">
    <img src="More_result/StableMofusion/base/03.gif" width="200" style="margin: 10px;">
    <img src="More_result/StableMofusion/base/02.gif" width="200" style="margin: 10px;">
  </div>
</div>

---

## Getting Started 🚀

### Using Free-T2M with **MDM**:

1. **Set Up the MDM Environment**:  
   Follow the [official MDM repository](https://github.com/GuyTevet/motion-diffusion-model) to configure the environment. Ensure all dependencies are installed.

2. **Integrate Free-T2M**:  
   Replace files in the MDM directory with those provided in this repository. This adds Free-T2M’s frequency-enhanced components.

3. **Train and Test**:  
   Use the original MDM training/testing pipelines. Free-T2M enhances robustness and precision during motion generation.

---

### Using Free-T2M with **StableMoFusion**:

1. **Set Up the StableMoFusion Environment**:  
   Refer to the [StableMoFusion repository](https://github.com/h-y1heng/StableMoFusion) for setup instructions.

2. **Integrate Free-T2M**:  
   Replace specific components in the StableMoFusion directory with files from this repository to enable Free-T2M’s enhancements.

3. **Train and Test**:  
   Use the StableMoFusion training/testing protocols. Free-T2M improves semantic alignment and motion quality.

---

## Contribute and Explore 🌟

Feel free to explore the repository and contribute to shaping the future of motion generation. Together, let’s redefine possibilities in text-to-motion generation! 🚀✨
