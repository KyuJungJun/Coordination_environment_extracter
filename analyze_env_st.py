#!/usr/bin/env python
# This code extracts the lithium environment of all of lithium sites provided in a structure file.
import os, sys
import numpy as np
import scipy
import argparse
from scipy.spatial import ConvexHull
from itertools import permutations
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import *
from pymatgen.core.composition import *
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.outputs import *
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import *

__author__ = "KyuJung Jun"
__version__= "0.1"
__maintainer__="KyuJung Jun"
__email__ = "kjun@berkeley.edu"
__status__="Development"

'''
Input for the script : path to the structure file supported by Pymatgen
Structures with partial occupancy should be ordered or modified to full occupancy by Pymatgen.
'''
parser = argparse.ArgumentParser()
parser.add_argument('structure', help='path to the structure file supported by Pymatgen', nargs='?')
parser.add_argument('envtype', help='both, tet, oct, choosing which perfect environment to reference to', nargs='?')
args = parser.parse_args()


class HiddenPrints:
    '''
    class to reduce the output lines
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def non_elements(struct, sp='Li'):
    '''
    struct : structure object from Pymatgen
    sp : the mobile specie
    returns the structure with all of the mobile specie (Li) removed
    '''
    num_li = struct.species.count(Element(sp))
    species = list(set(struct.species))
    species.remove(Element("O"))
    stripped = struct.copy()
    stripped.remove_species(species)
    stripped = stripped.get_sorted_structure(reverse=True)
    return stripped


def site_env(coord, struct, sp="Li", envtype='both'):
    '''
    coord : Fractional coordinate of the target atom
    struct : structure object from Pymatgen
    sp : the mobile specie
    envtype : This sets the reference perfect structure. 'both' compares CSM_tet and CSM_oct and assigns to the lower one.
    'tet' refers to the perfect tetrahedron and 'oct' refers to the perfect octahedron
    result : a dictionary of environment information
    '''
    stripped = non_elements(struct)
    with_li = stripped.copy()
    with_li.append(sp, coord, coords_are_cartesian=False, validate_proximity=False)
    with_li = with_li.get_sorted_structure()
    tet_oct_competition = []
    if envtype=='both' or envtype=='tet':
        for dist in np.linspace(1, 4, 601):
            neigh = with_li.get_neighbors(with_li.sites[0], dist)
            if len(neigh)<4:
                continue
            elif len(neigh)>4:
                break
            neigh_coords = [i.coords for i in neigh]
            with HiddenPrints():
                lgf = LocalGeometryFinder(only_symbols=["T:4"])
                lgf.setup_structure(structure=with_li)
                lgf.setup_local_geometry(isite=0, coords=neigh_coords)
            try:
                site_volume = ConvexHull(neigh_coords).volume
                tet_env_list = []
                for i in range(20):
                    tet_env = {'csm':lgf.get_coordination_symmetry_measures()['T:4']['csm'], 'vol':site_volume, 'type':'tet'}
                    tet_env_list.append(tet_env)
                tet_env = min(tet_env_list, key=lambda x:x['csm'])
                tet_oct_competition.append(tet_env)

            except Exception as e:
                print(e)
                print("This site cannot be recognized as tetrahedral site")
            if len(neigh)==4:
                break
    if envtype=='both' or envtype=='oct':
        for dist in np.linspace(1, 4, 601):
            neigh = with_li.get_neighbors(with_li.sites[0], dist)
            if len(neigh)<6:
                continue
            elif len(neigh)>6:
                break
            neigh_coords = [i.coords for i in neigh]
            with HiddenPrints():
                lgf = LocalGeometryFinder(only_symbols=["O:6"], permutations_safe_override=False)
                lgf.setup_structure(structure=with_li)
                lgf.setup_local_geometry(isite=0, coords=neigh_coords)
            try:
                site_volume = ConvexHull(neigh_coords).volume
                oct_env_list = []
                for i in range(20):
                    '''
                    20 times sampled in case of the algorithm "APPROXIMATE_FALLBACK" is used. Large number of permutations
                    are performed, but the default value in the function "coordination_geometry_symmetry_measures_fallback_random"
                    (NRANDOM=10) is often too small. This is not a problem if algorithm of "SEPARATION_PLANE" is used.
                    '''
                    oct_env = {'csm':lgf.get_coordination_symmetry_measures()['O:6']['csm'], 'vol':site_volume, 'type':'oct'}
                    oct_env_list.append(oct_env)
                oct_env = min(oct_env_list, key=lambda x:x['csm'])
                tet_oct_competition.append(oct_env)
                    
            except Exception as e:
                print(e)
                print("This site cannot be recognized as octahedral site")
            if len(neigh)==6:
                break

    if len(tet_oct_competition)==0:
        return {'csm':np.nan, 'vol':np.nan, 'type':'Non_'+envtype}
    elif len(tet_oct_competition)==1:
        return tet_oct_competition[0]
    elif len(tet_oct_competition)==2:
        csm1 = tet_oct_competition[0]
        csm2 = tet_oct_competition[1]
        if csm1['csm']>csm2['csm']:
            return csm2
        else:
            return csm1

def extract_sites(struct, sp="Li", envtype='both'):
    '''
    struct : structure object from Pymatgen
    envtype : 'tet', 'oct', or 'both'
    sp : target element to analyze environment

    '''
    envlist = []
    for i in range(len(struct.sites)):
        if struct.sites[i].specie != Element(sp):
            continue
        site = struct.sites[i]
        singleenv = site_env(site.frac_coords, struct, sp, envtype)
        envlist.append({'frac_coords':site.frac_coords, 'type':singleenv['type'], 'csm':singleenv['csm'], 'volume':singleenv['vol']})
    return envlist

def export_envs(envlist, sp='Li', envtype='both', fname=None):
    '''
    envlist : list of dictionaries of environment information
    fname : Output file name
    '''
    if not fname:
        fname = "extracted_environment_info"+"_"+sp+"_"+envtype+".dat"
    with open(fname, 'w') as f:
        f.write('Structure file path : '+ os.getcwd()+'/'+args.structure+'\n')

        f.write('List of environment information\n')
        f.write('Species : '+sp+"\n")
        f.write('Envtype : '+envtype+"\n")
        for index, i in enumerate(envlist):
            f.write("Site index "+str(index)+": "+str(i)+'\n')

struct = Structure.from_file(args.structure)
site_info = extract_sites(struct, envtype=args.envtype)
export_envs(site_info, envtype=args.envtype)