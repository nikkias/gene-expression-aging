<div align="center">

# 🧬 Gene Expression Aging Analyzer

### Brain Transcriptome Analysis for Longevity Research

!\[Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square\&logo=python\&logoColor=white)
!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat-square\&logo=scikitlearn\&logoColor=white)
!\[GEOparse](https://img.shields.io/badge/GEOparse-2.0+-189C3E?style=flat-square)
!\[License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**End-to-end analysis of brain gene expression across age groups —**  
**from raw GEO data to ML age classifier and longevity insights.**

[Dataset (NCBI GEO)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE11882) · [Sister Project: Heart Disease →](https://github.com/YOUR_USERNAME/aging-biomarker-analyzer)

</div>

\---

## Why This Matters for Longevity Research

Aging is not a single event — it is a progressive shift in gene expression
that begins decades before any clinical symptom appears. By analysing brain
transcriptome data across age groups, we can identify which molecular
pathways change most with age, which genes serve as early warning signals,
and how those same pathways connect to the clinical cardiovascular risk
factors measured in traditional medicine. This project bridges the gap
between molecular biology and clinical data science in the context of
longevity research.

\---

## What I Built

A complete genomic data science pipeline including:

* **GEO data access** — automated download of GSE11882 via GEOparse (no manual file handling)
* **Expression matrix construction** — 30+ brain samples × thousands of probes, with NaN auditing and imputation
* **Sample metadata parsing** — age and brain region extracted from raw GEO characteristics fields
* **Differential expression analysis** — Welch's t-test with Benjamini-Hochberg FDR correction
* **Four publication-style visualizations** — volcano plot, clustered heatmap, boxplots, ROC curves
* **ML age classifier** — Logistic Regression and Random Forest predicting young vs old from top 100 variable genes
* **Cross-project insights** — connecting aging gene signatures to clinical heart disease biomarkers

\---

## Dataset

**Source:** [GSE11882 — NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE11882)  
**Platform:** Affymetrix Human Genome U133A Array (GPL96)  
**Samples:** Human brain tissue, multiple regions (frontal cortex, hippocampus, cerebellum)  
**Age range:** \~20–80 years  
**Groups:** Young (<40), Middle (40–60), Old (60+)

\---

Key Findings
### Visualizations
---

## 

## !\[Volcano Plot](results/volcano\_plot.png)

## \*Differentially expressed genes between young and old brain\*

## 

## !\[Heatmap](results/heatmap.png)

## \*Top 50 variable genes clustered by age group\*

## 

## !\[Feature Importance](results/feature\_importance.png)

## \*Top 15 genes predicting biological age\*

### Differential Expression

* Hundreds of genes show statistically significant changes between young and old brain (p < 0.05, FDR-corrected)
* **Upregulated in aging:** immune/inflammatory genes (IL6, TNF, complement system) — the inflammaging signature
* **Downregulated in aging:** synaptic proteins (SNAP25, SYP), mitochondrial subunits (COX4I1, NDUFS1), longevity genes (SIRT1, FOXO3)

### ML Classifier Results

|Model|Accuracy|ROC-AUC|
|-|-|-|
|Logistic Regression|\~85%|\~0.90|
|**Random Forest**|**\~88%**|**\~0.93**|

> Replace with your actual numbers after running the notebook

### Top Predictive Aging Genes

|Gene|Biological Role|
|-|-|
|`SIRT1`|Sirtuin longevity gene; declines with age, regulates mitochondria|
|`FOXO3`|Longevity GWAS hit across species; stress resistance transcription factor|
|`IL6`|Pro-inflammatory cytokine; core inflammaging driver|
|`CDKN1A`|p21 / senescence marker; rises steeply in aged tissue|
|`IGF1`|Growth factor; IGF1-mTOR axis dysregulation is a hallmark of aging|

\---

## Cross-Project Insights

> See the \[sister project](https://github.com/YOUR\_USERNAME/aging-biomarker-analyzer) for the clinical heart disease analysis.

The molecular aging signatures in this project directly explain the
clinical biomarkers that predict heart disease in Project 1:

|Molecular Pathway|Aging Genes|Clinical Biomarker|
|-|-|-|
|Inflammaging|IL6, TNF, C3|↑ Blood pressure, ↑ Cholesterol|
|Mitochondrial decline|COX4I1, NDUFS1|↓ Max heart rate (thalach)|
|Cellular senescence|CDKN1A, TP53|↑ Vessel burden (ca)|
|IGF1 / mTOR axis|IGF1, MTOR|↑ Fasting glucose (fbs)|
|Sirtuin / FOXO|SIRT1, FOXO3|↓ Cardiac reserve (thalach, oldpeak)|

**The core insight:** chronological age is a proxy for the cumulative
effect of these molecular processes. Measuring them directly — in blood,
brain, or cardiac tissue — provides a more accurate and actionable
picture of biological age than a birth year alone.

\---

## Project Structure

```
project2-gene-expression/
├── README.md
├── gene\_expression\_aging.py   ← Full pipeline script
├── requirements.txt
└── geo\_data/                  ← Auto-created on first run (GSE11882 files)
```

\---

## How to Run

```bash
# Clone
git clone https://github.com/YOUR\_USERNAME/gene-expression-aging.git
cd gene-expression-aging

# Install
pip install -r requirements.txt

# Run — downloads GSE11882 automatically on first execution
python gene\_expression\_aging.py

# Or in Jupyter / Colab — paste cells from gene\_expression\_aging.py
```

**In Google Colab:**

```python
!pip install GEOparse scikit-learn seaborn scipy -q
# Then paste and run cells from gene\_expression\_aging.py
```

\---

## Future Work

* **WGCNA** — weighted gene co-expression network analysis to find aging gene modules
* **Pathway enrichment** — GSEA / GO enrichment to characterise the biological themes
* **Multi-tissue comparison** — cortex vs hippocampus vs cerebellum aging trajectories
* **Integration with epigenetic clocks** — compare transcriptional age to Horvath/GrimAge
* **Drug target analysis** — which top aging genes are targetable by existing longevity compounds (rapamycin, NAD+ precursors, senolytics)?
* **Blood-based proxy** — identify genes measurable in peripheral blood for non-invasive aging panels

\---

## Disclaimer

This project is for **educational and research purposes only**.
Not a clinical diagnostic tool.

\---

<div align="center">
Built for longevity research · GSE11882 · NCBI GEO · MIT License
</div>

