# Bachelor Thesis: Deep Learning–Based Risk Analysis of UK Biobank Atherosclerosis Data Using Electronic Health Records

This repository contains the code developed as part of my bachelor thesis at the **Heinig Lab** and is supervised by Korbinian Träuble
([Heinig Lab, Helmholtz Munich](https://www.helmholtz-munich.de/en/icb/research-groups/heinig-lab)). The project is conducted within the [Bioinformatics B.Sc. program](https://www.tum.de/studium/studienangebot/detail/bioinformatik-bachelor-of-science-bsc) at **<span style="color:#0065bd;">TUM</span>** (Technical University of Munich) and **<span style="color:#007c30;">LMU</span>** (Ludwig-Maximilians-University of Munich).

## Table of Content

1. [Reproduction of Georgakis Lab Results](#1-reproduction-of-georgakis-lab-results)

## 1. Reproduction of Georgakis Lab Results

As a first step, the results of the carotid plaque detection pipeline developed by the Georgakis Lab are reproduced using UK Biobank carotid ultrasound data.

The motivation for reproducing these results was

1. Due to data protection and access restrictions, the original results and intermediate outputs cannot be directly shared. Re-running the pipeline on the UK Biobank Research Analysis Platform is therefore necessary to obtain the corresponding outputs.

2. In addition to reproducing the original inference results, this reproduction step allows us to explicitly derive image-level plaque counts (0, 1, 2, or 3 detected plaques per image), which are required for further analyses and are not provided as a standalone output in the original setup.

The implementation follows the preprocessing and inference logic described in *Georgakis et al., Deep learning-based carotid plaque detection in population imaging*  and the authors’ reference notebook as closely as possible, while restructuring the workflow into standalone scripts suitable for large-scale execution on DNAnexus.
<p align="center">
  <img width="902" height="418" alt="image" src="https://github.com/user-attachments/assets/f78d7f12-75a0-4153-9a8e-ed55c3bfe175" />
</p>

**Figure 1:** Example carotid ultrasound image before and after automated plaque detection.
The left panel shows the preprocessed carotid ultrasound image, while the right panel shows the output of the deep learning–based plaque detection model, with bounding boxes indicating detected plaques and associated confidence scores.  
Figure adapted from the reference Jupyter notebook provided by Georgakis et al. and illustrates the overall functionality of the inference workflow.
