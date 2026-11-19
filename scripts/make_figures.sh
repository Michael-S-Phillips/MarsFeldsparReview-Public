#!/bin/bash
# Regenerate the spectral figures of the paper into output/ (run from the repository root).
#   Fig. 4            scripts/figures/plot_confidence_criteria_r3.py
#   Figs. 5-9         scripts/figures/plot_spectra_r2.py
#   Supp. Fig. 2      scripts/figures/plot_supp_fig2_ratio_cr.py
# FIG_THEME=light gives white-background versions (the paper uses the default charcoal theme).
set -e
cd "$(dirname "$0")/.."
mkdir -p output
python scripts/figures/plot_spectra_r2.py
python scripts/figures/plot_confidence_criteria_r3.py
python scripts/figures/plot_supp_fig2_ratio_cr.py output/supp_figure_2_ratio_cr.png
echo "figures written to output/"
