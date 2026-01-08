import numpy as np
import pandas as pd
import os 
from tqdm import tqdm
from pymatgen.io.cif import CifParser

from mofchecker import MOFChecker

import argparse

descriptors = ['has_carbon',
 'has_hydrogen',
 'has_atomic_overlaps',
 'has_overcoordinated_c',
 'has_overcoordinated_n',
 'has_overcoordinated_h',
 'has_undercoordinated_c',
 'has_undercoordinated_n',
 'has_undercoordinated_rare_earth',
 'has_metal',
 'has_lone_molecule',
 'has_high_charges',
 'has_suspicicious_terminal_oxo',
 'has_undercoordinated_alkali_alkaline',
 'has_geometrically_exposed_metal']

def main(args):
    gen_dir = args.res_path
    valid_structs = []
    pass_idx = []
    for i in tqdm(range(args.max_process)):
        try:
            pred_mof_path = os.path.join(gen_dir, f'{args.prefix}_{i}.cif')
            parser = CifParser(pred_mof_path)
            pred_structure = parser.get_structures()[0]
            valid_structs.append(pred_structure)
            pass_idx.append(i)
        except:
            pass

    mofchecker_dict = []
    for s in tqdm(valid_structs, desc="    MOFChecker"):
        try:
            mofchecker = MOFChecker(
                structure=s,
                symprec=None,
                angle_tolerance=None,
                primitive=False
            )
            desc = mofchecker.get_mof_descriptors(descriptors = descriptors)
            all_checks = []
            for k, v in desc.items():
                if type(v) == bool:
                    if k == "has_3d_connected_graph":
                        continue
                    if k in ["has_carbon", "has_hydrogen", "has_metal", "is_porous"]:
                        all_checks.append(int(v))
                    else:
                        all_checks.append(int(not v))
            desc["all_checks"] = np.all(all_checks)
        except: # SymmetryUndeterminedError or IndexError:
            # import ipdb; ipdb.set_trace()
            # all checks failed if PyMatGen leads to an error
            desc = {
                "has_carbon": False,
                "has_hydrogen": False,
                "has_atomic_overlaps": True,
                "has_overcoordinated_c": True,
                "has_overcoordinated_n": True,
                "has_overcoordinated_h": True,
                "has_undercoordinated_c": True,
                "has_undercoordinated_n": True,
                "has_undercoordinated_rare_earth": True,
                "has_metal": False,
                "has_lone_molecule": True,
                "has_high_charges": True,
                "has_suspicicious_terminal_oxo": True,
                "has_undercoordinated_alkali_alkaline": True,
                "has_geometrically_exposed_metal": True,
                "all_checks": False,
            }
        mofchecker_dict.append(dict(desc))


    mofchecker_pd = pd.DataFrame(
        mofchecker_dict,
        index = pass_idx,
        columns=[
            "has_carbon",
            "has_hydrogen",
            "has_atomic_overlaps",
            "has_overcoordinated_c",
            "has_overcoordinated_n",
            "has_overcoordinated_h",
            "has_undercoordinated_c",
            "has_undercoordinated_n",
            "has_undercoordinated_rare_earth",
            "has_metal",
            "has_lone_molecule",
            "has_high_charges",
            "has_suspicicious_terminal_oxo",
            "has_undercoordinated_alkali_alkaline",
            "has_geometrically_exposed_metal",
            "all_checks",
        ],
    )

    print(mofchecker_pd.mean().to_dict())
    val_path = os.path.join(os.path.dirname(args.res_path), 'validity.csv')
    mofchecker_pd.to_csv(val_path, index=True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', required=True)
    parser.add_argument('--max_process', default=1000, type=int)
    parser.add_argument('--prefix', default='pred', type=str)
    args = parser.parse_args()
    main(args)