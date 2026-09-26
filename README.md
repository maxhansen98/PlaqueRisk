# Deep Learning-Based Risk Analysis of UK Biobank Atherosclerosis Data

**Author:** Max-Malte Hansen
**Advisor:** M.Sc. Korbinian Träuble · **Supervisor:** Dr. Matthias Heinig

Code and thesis for a bachelor's project carried out at the
[Heinig Lab, Helmholtz Munich](https://www.helmholtz-munich.de/en/icb/research-groups/heinig-lab),
within the [Bioinformatics B.Sc. programme](https://www.tum.de/studium/studienangebot/detail/bioinformatik-bachelor-of-science-bsc)
of TUM and LMU Munich.

The work asks whether circulating proteins carry a detectable signature of subclinical carotid
atherosclerosis, and examines the imaging model that produces the phenotype such a search
depends on.

## Repository layout

The two blocks follow the division used in the thesis: the first works at cohort scale on UK
Biobank, the second on the external dataset Ar-PlaqSegm1.

| Path | Contents |
|------|----------|
| `block1_cohort/` | Phase 1 --- UK Biobank: cohort scan, pipeline reproduction, proteomics and health-record analyses |
| `block2_detection/` | Phase 2 --- Ar-PlaqSegm1: dataset construction, fine-tuning, evaluation and thesis figures |
| `common/` | Shared modules: `paths.py` resolves all data locations, `figlabel.py` holds the figure styling |
| `thesis/` | LaTeX sources, figures and bibliography |
| `analysis/`, `reference/` | Proteomics summaries and background PDFs, both excluded from version control |

Each block separates `scripts/` from `results/`, and `block2_detection/data/` holds the image
data and model checkpoints. Everything except the scripts is excluded from version control.

Data paths are resolved from the repository root by `common/paths.py`, so the scripts run
regardless of the working directory.

## Pipeline

**Cohort-scale inference.** `preprocess_ukb_files.py` extracts the image archives and writes the
manifest; `run_plaque_inference.py` applies the base detection model across it;
`merge_main_axis.py` aggregates the result into per-participant plaque counts. These stages run
on the UK Biobank Research Analysis Platform.

**Dataset construction.** `prepare_yolo_seg_dataset.py` converts the Ar-PlaqSegm1 masks into
polygon labels; `prepare_yolo_det_dataset.py` derives the corresponding bounding-box dataset used
for the detection control.

**Training.** `finetune_yolo_seg.py` and `finetune_yolo_det.py` fine-tune the segmentation and
detection models. `run_cv.py` provides cross-validation over the full development set.

**Evaluation.** `evaluate_seg_testset.py` evaluates a checkpoint with a confidence sweep;
`compare_variants.py` compares single models and pooled combinations; `threshold_on_val.py`
implements the reporting protocol, in which every operating point is selected on the validation
split; `rescore_instances.py` fits and evaluates the post-hoc instance re-scorer.

**Figures.** `make_appendix_gallery.py`, `make_qualitative_figure.py`,
`make_echogenicity_figure.py` and `make_pr_curves.py` regenerate the image figures of the thesis
from the data and the model checkpoints, so that every panel and every number printed on it is
recomputed rather than transcribed. `common/figlabel.py` holds the shared panel styling.

## Data

Ar-PlaqSegm1 is publicly available
([Mendeley Data, doi:10.17632/8srkpz52dy.1](https://doi.org/10.17632/8srkpz52dy.1)).

UK Biobank data are accessible only through the Research Analysis Platform under an approved
application and may not be redistributed. Nothing derived from individual-level UK Biobank data
is tracked in this repository; the corresponding paths are excluded in `.gitignore`.

## Environment

Python 3.12 with `ultralytics` 8.3.248, `torch` 2.9.1, `opencv-python` 4.12.0.88, `numpy` 2.2.6,
`pandas` 2.3.3, `scikit-learn` 1.9.0, `scipy` 1.17.1 and `matplotlib` 3.9.2. Segmentation models
were trained on Apple silicon using the Metal Performance Shaders backend; cohort-scale inference
ran on platform-provided cloud instances.

## Licence

MIT, see `LICENSE`.
