# 2DChiralPolACD
2D Chiral Polaritonics for ACD 

Code and scripts for reproducing 
Andrew H. Salij, Randall H. Goldsmith, Roel Tempelaar
Theory predicts 2D Chiral polaritons based on achiral Fabry-Perot cavities using apparent circular dichroism.

[![DOI](https://zenodo.org/badge/688635542.svg)](https://zenodo.org/doi/10.5281/zenodo.10152500)

To construct a suitable environment, create a conda environment
```bash
conda create --name chiralpol
conda activate chiralpol
```
then run
```bash
pip install numpy scipy pandas matplotlib sympy 
```
Install time: ~1 minute
Run time for all scripts: ~10 minutes 

Full environment and all package version numbers included at bottom of document.

The main folder contains all necessary files for core functionality and should be on PATH.

The Scripts folder contains scripts for calculations.

The Scripts\Figures folder contains figures and scripts for constructing figures. An example figure is shown here:
<figure>
<img src=https://github.com/andrewsalij/2DChiralPolACD/blob/main/Scripts/Figures/figure_vp_35_70_v5quad_hel_pol.png alt="Helical polaritonic characteristic comparison" width = "300px"/>
<figcaption align = "center">Comparison of helical polaritonic characteristics </figcaption>
</figure>

Calculations permit a variety of approximations to be made or omitted, with the corresponding functions detailing what 
precisely is calculated. 

In order, the figures are
Figure 1: Made in a vector graphics editor (not here)
Figure 2: chiral_int_triple_[version].pdf, made in figure_chiral_int_triple_plot.py w/ data construction in chiral_triple_scan.py
Figure 3: figure_vp_35_70_[version]quad_hel_pol.pdf, made in figure_quad_comparison.py w/ data construction in ljc_quad_comparison.py
Figure 4: figure_lp_comparison_quadchart_[version].pdf, made in figure_ljc_sweep_quad_comparison_v2.py w/ data construction in ljc_sweep_quad_comparison_data.py
Figure 5: fig_one_v_two_[version].pdf, made in fig_one_v_two_comparison.py w/ data construction in ljc_one_v_two_comparison.py
Figure 6: fig_di_bari_comparison_[version].png (molecule added in vector graphics editor in post), made in figure_di_bari_v2.py w/ data
construction in di_bari_reprod.py and di_bari_subset.py

Figure S1: figure_vp_35_70_[version]quad_hel_pol.pdf and rev_figure_vp_35_70_[version]quad_hel_pol.pdf, made in figure_quad_comparison.py w/ data from   ljc_quad_comparison.py under different settings


Note that there are rare instances of invalid values (divide by 0, mostly) in helicity construction and in tanh colormapping, but they
are not relevant for final results. 

This codebase is a partial clone of andrewsalij/SalijPhDWork (GPL 3.0), and it is released under GPL 3.0.
