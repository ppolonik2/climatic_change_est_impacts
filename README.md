This directory contains scripts to make the figures for:
# Estimating the impacts of climate change: reconciling disconnects between physical climate and statistical models

Link to manuscript in Climatic Change here (when avaiable)

Contact: ppolonik@ucsd.edu

Download data separately from the Harvard Dataverse.
   Due to file size limitations, data and gdp_data need to be downloaded separately
   gdp_data should be placed in ./data/gdp/
   
Link [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AXLJZ6)

## Directory descriptions:

```
   environment.yml contains the conda environment specifications used to run these scripts.
                   It is not necessarily a clean minimal version, but it works on our machine
                   to run all the scripts below.

   ./data/       should contain all data required to run scripts. Download separately and place here.

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
```


