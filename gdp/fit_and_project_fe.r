#!/usr/bin/env Rscript

library("fixest")
library("data.table")

#   Arguments: 
#       Path to CSV file used for the model
#       Formula of the form y ~ x+... | fixed effects
#           Example: outcome ~ Tf + Tf2 + Pf + Pf2 | id
#       (optional) file path to projection data with same var names as formula
#   
#   Returns:
#       CSV of coefficients and standard errors
#           Will be same name as infile with _fit appended
#       If optional projection path is included, also a CSV of original with proj column
#           Will overwrite input projection with same data, but new column outcome_proj

args = commandArgs(trailingOnly=TRUE)

infile = args[1]
formulastr = args[2]
weightvar = args[3]
if (length(args)>3){
    projfile = args[4]
}
print(infile)
outfile = paste0(strsplit(infile,'.csv')[1],'_fit','.csv')
fefile = paste0(strsplit(infile,'.csv')[1],'_fes','.csv')
texfile = paste0(strsplit(infile,'.csv')[1],'.tex')
texfefile = paste0(strsplit(infile,'.csv')[1],'_fes','.tex')

#   df = read.csv(infile)
df = fread(infile)

#   Change unit columns to factor for unit-time effects
#   if("pixel" %in% colnames(df))
#   {
#       df$pixel = as.factor(df$pixel)
#   }
#   if("admin_ix" %in% colnames(df))
#   {
#       df$admin_id = as.factor(df$admin_id)
#   }
#   if("country_id" %in% colnames(df))
#   {
#       df$country_id = as.factor(df$country_id)
#   }

#   cluster options: vcov='cluster','twoway'
if (weightvar!='None'){
    #   fit = feols(as.formula(formulastr), df, vcov='cluster',weights=df[weightvar]) 
    weightform = as.formula(paste0('~',weightvar))
    fit = feols(as.formula(formulastr), df, vcov='cluster',weights=weightform) 
} else {
    fit = feols(as.formula(formulastr), df, vcov='cluster') 
}

feformulastr = strsplit(formulastr,split='~',fixed=T)[[1]][2]
print(feformulastr)
feformulastr = strsplit(feformulastr,'|',fixed=T)[[1]][1]
print(feformulastr)
df_wFE = df
df_wFE$sumFE = fit$sumFE
FEfit = feols(as.formula(paste0('sumFE~',feformulastr)),df_wFE)

outtable = data.frame(coef(fit),se(fit),pvalue(fit))
names(outtable) <- c('coef','se','pval')

write.csv(outtable,outfile)
write(etable(fit, tex = TRUE),file=texfile)
write(etable(FEfit, tex = TRUE),file=texfefile)

#      Save fixed effects too
#   fedf = data.frame(fixef(fit)[1])
#   write.csv(fedf,fefile)
write.csv(data.frame(fit$sumFE),fefile)

#   Projections
if (length(args)>3){
    print('projecting')
    dfproj = fread(projfile)
    outcome_proj = predict(fit,dfproj)
    dfproj$outcome_proj = outcome_proj
    fwrite(dfproj,projfile)
}
