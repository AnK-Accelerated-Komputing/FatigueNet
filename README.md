# FatigueNet

This repo contains/will contain FatigueNet data and required tools/scripts to use it along with benchmarking code on different surrogate AI models and documents and tutorials related to dataset from generating similar data to exploting the data to its fullest.

## Current Progress
- [x] Generation of 1,200 3D FEA simulations for stepped and dog-bone shafts
- [x] Evaluation under pure tensile, pure torsional, and combined loading conditions
- [x] Node-wise fatigue life field extraction
- [x] Benchmarking with Pointwise MLP, Reg-DGCNN, and Transolver
- [ ] Tutorial generating raw data


## FatigueNet Dataset
The dataset compromises of structural fatigue life simulation outputs for unstructured 3D meshes of steel shafts. Simulations for 2 different shaft families (stepped shafts and dog-bone shafts) have been provided, each with 200 unique geometric variations sampled using Latin Hypercube Sampling (LHS). These 400 geometries were evaluated under 3 different loading conditions: pure tensile, pure torsional, and combined tension-torsion under fully reversed cyclic loading, making total 1,200 independent finite-element simulations. The dataset spans both high-cycle fatigue (HCF) and low-cycle fatigue (LCF) regimes and contains node-wise equivalent alternating stresses converted to crack-initiation life.

![Shafts Configuration](https://drive.google.com/uc?id=1gmlS-pTHh49p2mcPsv4fF6yl1jAJjISZ)
*Figure: Shafts configuration and geometric parameters used for dataset generation*

## Example Inference 
For proof of concept we have benchmarked different network architectures including Transolver, which utilizes global attention mechanisms, and obtained the following inference.\

![Fatigue Life Inference Result 1](https://drive.google.com/uc?id=1Jh5UKz92h47D6KhufONOj3VhIsmHfk1d)
*Figure: Spatial visualization of node-wise fatigue life prediction of stepped shaft under combined loading*


![Fatigue Life Inference Result 2](https://drive.google.com/uc?id=1YEiEN9omaa044lOejEGuuOLFqP1ovH0m)
*Figure: Spatial visualization of node-wise fatigue life prediction of dog-bone shaft under combined loading*

## Citation
If you use this dataset in your work, please consider citing the following publications.
```bibtex
@inproceedings{FATIGUENET2026,
  author    = {Sachin Saud and Bipsan Nepal and Bipin Shrestha and Rachit Rijal and Susil Chhetri and Tulsi Narayan Shrestha and Amit Regmi and Akio Tanaka},
  title     = {{FATIGUENET: DEEP LEARNING SURROGATE MODEL FOR NODE-WISE FATIGUE LIFE PREDICTION IN 3D STEEL SHAFTS}},
  booktitle = {Proceedings of the ASME 2026 International Design Engineering Technical Conferences
               and Computers and Information in Engineering Conference (IDETC-CIE 2026)},
  year      = {2026},
  address   = {Houston, TX, USA},
  paperid   = {DETC2026-193892},
  publisher = {American Society of Mechanical Engineers (ASME)}
}
```
