import os, re

# ---- sostituisce caratteri proibiti con '___'
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|&]', '___', name)

def rename_recursively(root_dir: str) -> int:
    """
    Visita ricorsivamente root_dir e rinomina tutti i file
    contenenti caratteri proibiti. Ritorna il conteggio rinominati.
    """
    renamed = 0
    for dirpath, _, files in os.walk(root_dir):
        for fname in files:
            new_fname = sanitize_filename(fname)
            if fname != new_fname:
                src = os.path.join(dirpath, fname)
                dst = os.path.join(dirpath, new_fname)
                os.rename(src, dst)
                renamed += 1
    return renamed

# ------------------- MAIN -------------------------------------------
dataset_root = os.getcwd()

# 1) train
train_dir = os.path.join(dataset_root, "ccpd_subset_base", "train")
renamed_train = rename_recursively(train_dir)
print(f" Rinominate {renamed_train} immagini in '{train_dir}'")

# 2) tutte le sottocartelle di test
test_root = os.path.join(dataset_root, "ccpd_test")
if os.path.isdir(test_root):
    renamed_test = rename_recursively(test_root)
    print(f" Rinominate {renamed_test} immagini in '{test_root}'")
else:
    print(f" Cartella test non trovata: {test_root}")
