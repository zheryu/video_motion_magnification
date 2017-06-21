"""
Complex steerable filter utility functions

Python version of pyrToolsExt.

This is a direct reimplementation of utility functions used in the original Phase-based Motion Magnification
project authored by Wadhwa et al.
"""
import math
from numpy import *

def simplify_phase(x):
    """
    Moves x into the [-pi, pi] range.
    """
    return ((x + pi) % (2*pi)) - pi

def max_scf_pyr_height(dims):
	"""
	Gets the maximum possible steerable pyramid height
    dims: (h, w), the height and width of your desired filters in a tuple
	"""
	return int(log2(min(dims[:2]))) - 2

def get_polar_grid(dims):
    center = ceil((array(dims))/2).astype(int)
    xramp, yramp = meshgrid(linspace(-1, 1, dims[1]+1)[:-1], linspace(-1, 1, dims[0]+1)[:-1])
    
    theta = arctan2(yramp, xramp)
    r = sqrt(xramp**2 + yramp**2)
    
    # eliminate the zero at the center
    r[center[0], center[1]] = min((r[center[0], center[1]-1], r[center[0]-1, center[1]]))/2
    return theta, r

def get_angle_mask_smooth(index, num_bands, angle, is_complex):
    order = num_bands-1
    const = sqrt((2**(2*order))*(math.factorial(order)**2)/(num_bands*math.factorial(2*order)))
    angle = simplify_phase(angle+(pi*index/num_bands))
    
    if is_complex:
        return const*(cos(angle)**order)*(abs(angle) < pi/2)
    else:
        return abs(sqrt(const)*(cos(angle)**order))

def get_filters_smooth_window(dims, orientations, cos_order=6, filters_per_octave=6, is_complex=True, pyr_height=-1):
    """
    A complex steerable filter generator with a smoother window. Better for quarter octave or half octave decompositions.
    """
    max_pyr_height = max_scf_pyr_height(dims)
    if pyr_height == -1 or pyr_height > max_pyr_height:
        pyr_height = max_pyr_height
    total_filter_count = filters_per_octave * pyr_height
    
    theta, r = get_polar_grid(dims)
    r = (log2(r) + pyr_height)*pi*(0.5 + (total_filter_count / 7)) / pyr_height
    
    window_function = lambda x, c: (abs(x - c) < pi/2).astype(int)
    compute_shift = lambda k: pi*(k/(cos_order+1)+2/7)
    
    rad_filters = []
    
    total = zeros(dims)
    a_constant = sqrt((2**(2*cos_order))*(math.factorial(cos_order)**2)/((cos_order+1)*math.factorial(2*cos_order)))
    for k in range(total_filter_count):
        shift = compute_shift(k+1)
        rad_filters += [a_constant*(cos(r-shift)**cos_order)*window_function(r,shift)]
        total += rad_filters[k]**2
    rad_filters = rad_filters[::-1]
    
    center = ceil(array(dims)/2).astype(int)
    low_dims = ceil(array(center+1.5)/4).astype(int)
    total_cropped = total[center[0]-low_dims[0]:center[0]+low_dims[0]+1, center[1]-low_dims[1]:center[1]+low_dims[1]+1]
    
    low_pass = zeros(dims)
    low_pass[center[0]-low_dims[0]:center[0]+low_dims[0]+1, center[1]-low_dims[1]:center[1]+low_dims[1]+1] = abs(sqrt(1+0j-total_cropped))
    total += low_pass**2    
    high_pass = abs(sqrt(1+0j-total))
    
    anglemasks = []
    for i in range(orientations):
        anglemasks += [get_angle_mask_smooth(i, orientations, theta, is_complex)]

    out = [high_pass]
    for i in range(len(rad_filters)):
        for j in range(len(anglemasks)):
            out += [anglemasks[j]*rad_filters[i]]
    out += [low_pass]
    return out

def get_radial_mask_pair(r, rad, t_width):
    log_rad = log2(rad)-log2(r)
    hi_mask = abs(cos(log_rad.clip(min=-t_width, max=0)*pi/(2*t_width)))
    lo_mask = sqrt(1-(hi_mask**2))
    return (hi_mask, lo_mask)

def get_angle_mask(b, orientations, angle):
    order = orientations - 1
    a_constant = sqrt((2**(2*order))*(math.factorial(order)**2)/(orientations*math.factorial(2*order)))
    angle2 = simplify_phase(angle - (pi*b/orientations))
    return 2*a_constant*(cos(angle2)**order)*(abs(angle2) < pi/2)

def get_filters(dims, r_vals=None, orientations=4, t_width=1):
    """
    Gets a steerbale filter bank in the form of a list of ndarrays
    dims: (h, w). Dimensions of the output filters. Should be the same size as the image you're using these to filter
    r_vals: The boundary between adjacent filters. Should be an array.
        e.g.: 2**np.array(list(range(0,-7,-1)))
    orientations: The number of filters per level
    t-width: The falloff of each filter. Smaller t_widths correspond to thicker filters with less falloff
    """
    if r_vals is None:
        r_vals = 2**np.array(list(range(0,-max_scf_pyr_height(dims)-1,-1)))
    angle, r = get_polar_grid(dims)
    hi_mask, lo_mask_prev = get_radial_mask_pair(r_vals[0], r, t_width)
    filters = [hi_mask]
    for i in range(1, len(r_vals)):
        hi_mask, lo_mask = get_radial_mask_pair(r_vals[i], r, t_width)
        rad_mask = hi_mask * lo_mask_prev
        for j in range(orientations):
            angle_mask = get_angle_mask(j, orientations, angle)
            filters += [rad_mask*angle_mask/2]
        lo_mask_prev = lo_mask
    filters += [lo_mask_prev]
    return filters