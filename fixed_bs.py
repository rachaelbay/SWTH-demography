#! /bin/env python
import numpy
from numpy import array
import dadi

#import demographic models
import swainsons_models

##Define initial conditions
outFile=open("/Users/Rachael/Documents/MigratoryBirds/Swainsons/Demography/output/multiple_2000/Z_fixed_bs.txt","w")
nuI_init=10
nuC_init=5
m12_init=1
m21_init=0.1
T_init=0.3375


for i in xrange(1,101):
	print i,'\n'
	# Parse the SNP file to generate the data dictionary
	filename = str('/Users/Rachael/Documents/MigratoryBirds/Swainsons/Demography/formatdata/BS_multipleSNP/ZBS/BS2000_' + str(i) + str('.txt'))
	dd = dadi.Misc.make_data_dict(filename)

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
	func = swainsons_models.fixed_split_migration
	params = array([nuI_init, nuC_init, m12_init, m21_init, T_init])
	upper_bound = [100, 100, 10, 10, 10]
	lower_bound = [1e-5, 1e-5, 0, 0, 0]
	
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
	print 'Theta:',theta,"\n"
	
	p0 = dadi.Misc.perturb_params(params, fold=1, lower_bound=lower_bound, upper_bound=upper_bound)
	print 'perturbed parameters: ', p0, "\n"
	popt = dadi.Inference.optimize_log(p0, fs, func_ex, pts, upper_bound=upper_bound, lower_bound=lower_bound, maxiter=None, verbose=len(params),fixed_params=[None,None,None,None,0.3984]) 
	print 'Optimized parameters:', repr(popt), "\n"
	
	#use the optimized parameters in a new model to try to get the parameters to converge
	new_model = func_ex(popt, ns, pts)
	ll_opt = dadi.Inference.ll_multinom(new_model, fs)
	print 'Optimized log-likelihood:', ll_opt, "\n"
	new_theta = dadi.Inference.optimal_sfs_scaling(new_model, fs)
	print 'Optimized theta:',new_theta,"\n"
	
	
	# Write the parameters and log-likelihood to the outFile
	#s = str(nuC_init) + '\t' + str(nuI_init) + '\t' + str(T_init) + '\n'
	s = str(i) + '\t' + str(new_theta) + '\t'
	s += str(ll_opt) + '\t'
	for i in range(0, len(popt)):
	    s += str(popt[i]) + '\t'
	s += '\n'
	outFile.write(s)


