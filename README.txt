
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #


This directory contains the scripts required to create the figures in:

Estimating the impacts of climate change: reconciling disconnects between physical climate and statistical models
Pascal Polonik, Katharine Ricke, Jennifer Burney
Climatic Change. 2025.


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

Directory descriptions:

   environment.yml contains the conda environment specifications used to run these scripts.
                   It is not necessarily a clean minimal version, but it works on our machine
                   to run all the scripts below.

   ./data/       contains all data required to run scripts. Download separately.

   ./figures/    contains all figures. Scripts save to here.

   ./trends/     contains scripts pertaining to climate trends

       interannual_vs_decadal_metrics_lens.py  Calculates shocks and shifts
       shock_vs_shift_lens.py                  Code for figure 7

   ./stations/   contains scripts pertaining to use of station data
   
       summary_figure.py                       Code for figure 1

   ./idealized/  contains scripts pertaining to idealized modeling

       cires_err.py                            Code for figure 2
       lineartestfunc_plusquad.py              Code for figure 4
       compare_Pomit_and_Perr_marginal.py      Code for figure 5
       quadoutcome.py                          Required for idealized modeling

   ./gdp/        contains scripts pertaining to GDP regressions and projections

       project_growth_func.py                  Code for figure 3 and figure 6
       means_project.py                        Code for figure 8
       get_gdp_fes.py                          Executes fixed effect GDP regressions and saves output
       fit_and_project_fe.r                    R script to run regressions


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #



