from pathlib import Path
from functools import partial
import argparse
import json

from pymatgen.io.cif import CifWriter

from mofdiff.common.relaxation import lammps_relax
from mofdiff.common.mof_utils import save_mofid, mof_properties
from mofid.id_constructor import extract_topology

from p_tqdm import p_umap


def main(input_dir, max_natoms=2000, get_mofid=True, ncpu=96):
    """
    max_natoms: maximum number of atoms in a MOF primitive cell to run zeo++/mofid.
    """
    all_files = list((Path(input_dir) / "cif").glob("*.cif"))

    save_dir = Path(input_dir) / "relaxed"
    save_dir.mkdir(exist_ok=True, parents=True)

    def relax_mof(ciffile):
        name = ciffile.parts[-1].split(".")[0]
        try:
            struct, relax_info = lammps_relax(str(ciffile), str(save_dir))
        except TimeoutError:
            return None

        if struct is not None:
            struct = struct.get_primitive_structure()
            CifWriter(struct).write_file(save_dir / f"{name}.cif")
            relax_info["natoms"] = struct.frac_coords.shape[0]
            relax_info["path"] = str(save_dir / f"{name}.cif")
            return relax_info
        else:
            return None

    results = p_umap(relax_mof, all_files, num_cpus=ncpu)
    relax_infos = [x for x in results if x is not None]
    with open(save_dir / "relax_info.json", "w") as f:
        json.dump(relax_infos, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    args = parser.parse_args()
    main(args.input)