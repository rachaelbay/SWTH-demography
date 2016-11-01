"""
Dadi demographic models for Swainson's Thrush
"""
import numpy
import dadi

#one ancestral population splits at time T ago with no migration
def bottleneck_split(params, (n1,n2), pts):
    nuI, nuC, T = params
    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

    phi = dadi.Integration.two_pops(phi, xx, T, nuI, nuC)

    model_sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return model_sfs


#one ancestral population splits at time T ago, then exchanges m12 and m21 proportions of migrants
def split_migration(params, (n1,n2), pts):
    nuI, nuC, T, m12, m21 = params
    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

    phi = dadi.Integration.two_pops(phi, xx, T, nuI, nuC, m12=m12, m21=m21)

    model_sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return model_sfs 
    
#with fixed theta: one ancestral population splits at time T ago, then exchanges m12 and m21 proportions of migrants
def fixed_split_migration(params, (n1,n2), pts):
    theta1 = 73.15
    nuI, nuC, T, m12, m21 = params
    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx,theta0=theta1)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

    phi = dadi.Integration.two_pops(phi, xx, T, nu1=nuI, nu2=nuC, m12=m12, m21=m21,theta0=theta1)

    model_sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return model_sfs     
