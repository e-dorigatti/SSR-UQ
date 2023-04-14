Frequentist Uncertainty Quantification in Semi-Structured Neural Networks
===

This is the source code of the AISTATS paper 

> Emilio Dorigatti, Benjamin Schubert, Bernd Bischl, David Ruegamer, Frequentist Uncertainty Quantification in Semi-Structured Neural Networks, *Proceedings of The 26th International Conference on Artificial Intelligence and Statistics*, PMLR 206:1924-1941, 2023. 

https://proceedings.mlr.press/v206/dorigatti23a.html

## Reproducibility

The folder `theory-simulations` contains the code to reproduce the simulations and figures 2-6 of the paper.
To generate figures 4-6, first execute `poisson_sddr_run.R` to train the neural networks, then the respective script for the figure.

The folder `skin-lesion` contains the code to reproduce the practical application on the skin lesion dataset:
 1. Download the dataset into the `skin-lesion/data` folder: https://challenge2020.isic-archive.com/
    - Choose `Download metadata v2 (2MB)`
    - Choose `Download DICOM Corrected* (23.0GB)` and unzip the images into `data/train`
 2. Run the first data-preparation script: `Rscript preprocess-1.R`
 3. Run the second data-preparation script: `python preprocess-2.py`
 4. Run cross-validation and grid search: `bash train.sh`
 5. Aggregate the predictions `python run.py aggregate`
 6. Fit GAMM models on the networks: `bash fit-gamm.sh`
 7. Finally, analyze their predictions: `Rscript analyze-gamm.R`

## Requirements

The theory simulations with the neural networks use the [R interface to Keras](https://keras.rstudio.com/), while the skin lesion example uses pytorch and [pytorch-lightning](pytorch-lightning.readthedocs.io/).