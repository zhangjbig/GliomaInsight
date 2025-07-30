# GliomaInsight

**GliomaInsight** is a visual analysis platform for glioma, designed to assist clinical decision-making by integrating multimodal MRI data and radiomics-based predictions. It supports interactive visualization, radiomics feature extraction, gene expression prediction, and TAM (tumor-associated macrophage) subtype infiltration analysis, especially focusing on Angio-TAMs, which are closely related to angiogenesis and patient prognosis.

---

## Key Features

### MRI Visualization

* Supports importing original images and tumor segmentation masks.
* Simultaneously loads 4 clinical MRI modalities: **Flair, T1, T1CE, T2**.
* Provides:

  * **3D view** to visualize tumor invasion in brain tissue.
  * **2D multi-planar view** with XYZ sliders.
* Includes **6 medical colormaps** (`bone`, `gist_ncar_r`, `hot_r`, `bone_r`, `jet`, `spectral_r`) to enhance contrast and tumor boundary clarity.

### Radiomics Feature Extraction

* Integrated **Pyradiomics** module for texture feature extraction from T2-weighted MRI.
* Supports multiple configurations:

  * Common Setting
  * CT
  * MR with 3mm / 5mm slice thickness
  * MR without resampling
* Offers customizable preprocessing and feature selection:

  * First-order statistics
  * Shape-based (3D) features
  * Five texture families: GLCM, GLRLM, GLSZM, GLDM, NGTDM

### DLP Analysis
- Swich to the DlpFeature and choose the Net type you need
- Use batch import to extract the deeplearning features.
- The extracting procedure and related data will be printed in the UI and your saving path.

### RadioML
This is a deep learning-based platform for predicting immunogenicity of T-cell activated antigens.
- This model is to identify prognostic radiomics features using clinical information and radiomics features extracted from T2-weighted MRI imagings.

### Gene Expression Prediction

* **Gene-Exp** module focuses on 7 glioma-related genes:
  `CD40`, `TRAM2`, `CCRL2`, `FCGR2A`, `JUNB`, `MBOAT1`, and `KYN`.
* Upload extracted radiomic features to estimate gene expression.
* Predict expression distribution in the context of the training cohort.
* Visualizes candidate **targeted drugs** associated with the predicted gene status.

### Angio-TAMs Enrichment Estimation

* Upload radiomics features to predict Angio-TAMs infiltration.
* Outputs enrichment score overlaid on population distribution.
* Stratifies **survival risk** based on TAM infiltration.
* Provides a noninvasive biomarker for **anti-angiogenic or immunotherapy decisions**.

---

## Installation

Clone the repository and navigate to the project folder:

```bash
git clone https://github.com/yourname/YourProject.git
cd YourProject
```

---

## Getting Started

* All required Python packages are listed in the `requirements.txt` file.
To install them, run the following command in your terminal:
```bash
pip install -r requirements.txt
```

Launch the GUI:

```bash
python mainCode.py
```

### Basic Radiomics Analysis (Pyradiomics)

* Click `Input Label` and select the segmentation mask (ROI).
* Click `Input Image` and choose your brain MRI image.
* Choose `Pyradiomics` and select settings (use `Common Settings` if unsure).
* Choose an output table path and file name.
* Click `Apply` to view and extract features.
* Click `Save Data` to store the results.

### Custom Radiomics Workflow

* In the `Pyradiomics` module:

  * Use the `Custom Settings` panel to select your data type.
  * Use `Select Features` to choose the features to keep.
  * Name your output file and click `Apply` to extract features.

### Prediction Modules: Gene-Exp & Angio-TAMs

* First upload the radiomics feature `.csv` file.
* Then click `Predict` to obtain gene expression or Angio-TAMs enrichment results.

---

## Example Data

You can use the sample folder `exampleSettings/BraTS_00495` to test the software and simulate the workflow.

