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

Free-T2M refines motion generation through a staged denoising process. The table below illustrates the transition from noise to high-quality motion at different denoising steps:

| **Denoising Step** | **Visualization**                               |
|--------------------|------------------------------------------------|
| Step 50(Noise)     | <img src="Visualization/Noise/21_50.gif" width="200"> |
| Step 40            | <img src="Visualization/Noise/21_40.gif" width="200"> |
| Step 30            | <img src="Visualization/Noise/21_30.gif" width="200"> |
| Step 20            | <img src="Visualization/Noise/21_20.gif" width="200"> |
| Step 10            | <img src="Visualization/Noise/21_10.gif" width="200"> |
| Step 0             | <img src="Visualization/Noise/21_0.gif" width="200"> |

---

This table provides a clear step-by-step visualization of the denoising process, from pure noise to refined motion. Let me know if additional details are needed!
## Visual Comparisons 🎬

### MDM Baseline vs. Free-T2M

The following table compares Free-T2M with the MDM baseline. Free-T2M produces more realistic and semantically aligned motions:

| **Ours**                                       | **Baseline**                                   |
|-----------------------------------------------|-----------------------------------------------|
| <img src="More_result/MDM/Ours/00.gif" width="200"> | <img src="More_result/MDM/base/00.gif" width="200"> |
| <img src="More_result/MDM/Ours/01.gif" width="200"> | <img src="More_result/MDM/base/01.gif" width="200"> |
| <img src="More_result/MDM/Ours/02.gif" width="200"> | <img src="More_result/MDM/base/02.gif" width="200"> |

---

### StableMoFusion Baseline vs. Free-T2M

The table below highlights the superior performance of Free-T2M over the StableMoFusion baseline, showcasing more realistic and semantically aligned motions:

| **Ours**                                       | **Baseline**                                   |
|-----------------------------------------------|-----------------------------------------------|
| <img src="More_result/StableMofusion/ours/04.gif" width="200"> | <img src="More_result/StableMofusion/base/04.gif" width="200"> |
| <img src="More_result/StableMofusion/ours/03.gif" width="200"> | <img src="More_result/StableMofusion/base/03.gif" width="200"> |
| <img src="More_result/StableMofusion/ours/02.gif" width="200"> | <img src="More_result/StableMofusion/base/02.gif" width="200"> |
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
