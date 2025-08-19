from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import json

import biotite.sequence.io.fasta as fasta
from biotite.structure import filter_canonical_amino_acids, get_residues, get_chains
from biotite.structure.io.pdb import PDBFile
from tap import tapify

AA_3_TO_1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "SEC": "U",
    "PYL": "O",
}


def pdb_to_sequence(pdb_path: Path) -> dict[str, dict[str, str]]:
    """
    Extracts the amino acid sequences for each chain in a PDB file.

    Args:
        pdb_path (Path): Path to the PDB file.

    Returns:
        dict[str, dict[str, str]]: Dictionary mapping complex IDs to their amino acid sequences (1-letter code).
    """
    # Read the PDB file and get the structure object
    structure = PDBFile.read(pdb_path).get_structure()
    complex_id = pdb_path.stem.split("-")[0]
    pdb_id, ab_chain, ag_chain, mutation = complex_id.split("_")
    mutation_res_id = int(mutation[2:-1])
    chain_with_mutation = mutation[1]
    # Use only the first model (in case of multiple models)
    structure = structure[0]
    structure = structure[filter_canonical_amino_acids(structure)]
    chains = get_chains(structure)
    chain_to_info = {}
    for chain in chains:
        dict_key = f"{complex_id}_{chain}"
        chain_atoms = structure[structure.chain_id == chain]
        chain_res_ids, chain_res_names = get_residues(chain_atoms)
        # Convert 3-letter residue names to 1-letter codes and join into a sequence string
        chain_sequence = "".join(AA_3_TO_1[res_name] for res_name in chain_res_names)
        chain_to_info[dict_key] = {"sequence": chain_sequence}
        if chain == chain_with_mutation:
            mutation_res_id = list(chain_res_ids).index(mutation_res_id)
            chain_to_info[dict_key]["mutation_index"] = mutation_res_id
    return chain_to_info


def convert_pdb_to_fasta(pdb_dir: Path, save_path: Path) -> None:
    """
    Converts all PDB files in a directory to a single JSON file containing all chain sequences.

    Args:
        pdb_dir (Path): Directory containing PDB files.
        save_path (Path): Path to save the resulting JSON file.
    """
    # Get a list of all PDB file paths in the directory
    pdb_paths = list(pdb_dir.glob("*.pdb"))
    with Pool() as pool:
        chain_to_sequence_dicts = {}
        for chain_sequence_dict in tqdm(
            pool.imap(pdb_to_sequence, pdb_paths), total=len(pdb_paths)
        ):
            chain_to_sequence_dicts.update(chain_sequence_dict)

    # save dictionary as json
    with open(save_path, "w") as f:
        json.dump(chain_to_sequence_dicts, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    tapify(convert_pdb_to_fasta)
