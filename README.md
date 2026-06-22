# AI-robotics-proteomics

Code repository accompanying the preprint:

Robotic perturbation proteomics and AI agents enable scalable drug mechanism discovery

Yuming Jiang, Cameron S. Movassaghi, Jesús Muñoz-Estrada, Niveda Sundararaman,
Amanda Momenzadeh, Jesse G. Meyer

bioRxiv (2026)

Preprint:
https://www.biorxiv.org/content/10.64898/2026.05.04.722718v1

This repository contains analysis notebooks, scripts, AI-generated outputs,
manual-check results, clinical validation analysis, figure-generation code,
and web-portal code associated with a large-scale HepG2 drug perturbation
proteomics study.

STUDY OVERVIEW

This project develops an end-to-end platform that combines semi-automated sample
preparation, rapid LC-MS/MS proteomics, and AI-agent-assisted data analysis for
scalable drug mechanism discovery.

In the associated study, 172 compounds were profiled in HepG2 cells with six
biological replicates per compound, generating 1,232 proteomes and quantifying
8,703 proteins.

REPOSITORY STRUCTURE

AI results/             AI-generated analysis outputs
ClinicalValidation/     EHR validation analyses
Figures/                Figure-generation scripts
MultiAgent/             GPT-based multi-agent interpretation workflow
data/                   Example and processed data
manual check/           Human-verified outputs
website/                Streamlit portal

MAIN ANALYSIS COMPONENTS

1. Proteomics data processing and quality control
- Protein filtering
- Sample filtering
- Missing-value assessment
- Missing-value imputation
- Batch correction
- QC visualization

2. Differential protein regulation analysis
- Welch's t-test
- Benjamini-Hochberg FDR correction
- Significant proteins:
  FDR < 0.05
  |log2FC| > 0.5

3. Pathway and mechanism-of-action analysis
- Hallmark pathways
- GO Biological Process pathways
- GSEA analysis
- Drug clustering
- Mechanism inference

4. AI-agent-assisted interpretation
- Molecular identity profiles
- Drug-level summaries
- Pathway interpretation
- Known MoA recovery
- Novel hypothesis generation
- Validation recommendations

5. Multi-agent framework
- Analyzer
- Reviewer
- Supervisor

6. Clinical validation
Includes EHR-based validation workflows.

7. Interactive website
https://drug-ai-robotics-proteomics.streamlit.app/

DATA AVAILABILITY

MassIVE accession:
MSV000101671

FTP:
ftp://massive-ftp.ucsd.edu/v12/MSV000101671/

SOFTWARE REQUIREMENTS

Python 3.13.9

Common dependencies:
numpy
pandas
scipy
statsmodels
scikit-learn
matplotlib
seaborn
gseapy
networkx
streamlit

BASIC USAGE

git clone https://github.com/xomicsdatascience/AI-robotics-proteomics.git

NOTES ON AI-GENERATED CODE

Some code and outputs were generated or assisted by AI systems.
All major analytical logic and outputs were manually reviewed.

CITATION

Jiang Y, Movassaghi CS, Muñoz-Estrada J, Sundararaman N,
Momenzadeh A, Meyer JG.

Robotic perturbation proteomics and AI agents enable scalable drug mechanism discovery

bioRxiv 2026.

DOI: 10.64898/2026.05.04.722718

CONTACT

Yuming Jiang
yuming.jiang@csmc.edu

Jesse G. Meyer
jesse.meyer@csmc.edu




