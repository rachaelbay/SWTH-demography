#! /bin/env python
import numpy
from numpy import array
import dadi

#import demographic models
import swainsons_models

##Define initial conditions
outFile=open("/Users/Rachael/Documents/MigratoryBirds/Swainsons/Demography/output/noisland_nomigb.txt","w")
nuC_init=1
nuI_init=1
T_init=1


# Parse the SNP file to generate the data dictionary
dd = dadi.Misc.make_data_dict('/Users/Rachael/Documents/MigratoryBirds/Swainsons/Demography/formatdata/noisland_dadi.txt')

# Extract the spectrum for ['YRI','CEU'] from that dictionary, with both
fs = dadi.Spectrum.from_data_dict(dd, pop_ids=['Inland','Coastal'], projections=[15,15], polarized=False)

#for i in xrange(0,10):
#fs= fs.sample()
ns = fs.sample_sizes
print 'sample sizes:', ns

# These are the grid point settings will use for extrapolation.
pts = [20,30,40]
# suggested that the smallest grid be slightly larger than the largest sample size. But this may take a long time.              

# bidirectional migration model
func = swainsons_models.bottleneck_split
params = array([nuC_init, nuI_init, T_init])
upper_bound = [100, 100, 10]
lower_bound = [1e-5, 1e-5, 0]

# Make the extrapolating version of the demographic model function.
func_ex = dadi.Numerics.make_extrap_func(func)
# Calculate the model AFS
model = func_ex(params, ns, pts)        
# Calculate likelihood of the data given the model AFS

# Likelihood of the data given the model AFS.
ll_model = dadi.Inference.ll_multinom(model, fs)
print 'Model log-likelihood:', ll_model, "\n"
# The optimal value of theta given the model.
theta = dadi.Inference.optimal_sfs_scaling(model, fs)

p0 = dadi.Misc.perturb_params(params, fold=1, lower_bound=lower_bound, upper_bound=upper_bound)
print 'perturbed parameters: ', p0, "\n"
popt = dadi.Inference.optimize_log(p0, fs, func_ex, pts, upper_bound=upper_bound, lower_bound=lower_bound, maxiter=None, verbose=len(params)) 
print 'Optimized parameters:', repr(popt), "\n"

#use the optimized parameters in a new model to try to get the parameters to converge
new_model = func_ex(popt, ns, pts)
ll_opt = dadi.Inference.ll_multinom(new_model, fs)
print 'Optimized log-likelihood:', ll_opt, "\n"
new_theta = dadi.Inference.optimal_sfs_scaling(new_model, fs)
print 'Optimized theta:',new_theta,"\n"


# Write the parameters and log-likelihood to the outFile
s = str(new_theta) + '\t'
s += str(ll_opt) + '\t'
for i in range(0, len(popt)):
    s += str(popt[i]) + '\t'
s += '\n'
outFile.write(s)

