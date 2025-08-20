import torch

# Amino acid constants
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_ALPHABET_SET = set(AA_ALPHABET)
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ALPHABET)}
GLYCINE_LINKER_LENGTH = 25


def load_esm_model(
    esm_model: str = "esm2_t33_650M_UR50D", hub_dir: str | None = None
) -> tuple:
    """Load an ESM2 model and batch converter.

    :param esm_model: Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm.
    :param hub_dir: Path to directory where torch hub models are saved.
    :return: A tuple of a pretrained ESM2 model and a BatchConverter for preparing protein sequences as input.
    """
    if hub_dir is not None:
        torch.hub.set_dir(hub_dir)
    model, alphabet = torch.hub.load("facebookresearch/esm:main", esm_model)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, alphabet, batch_converter
