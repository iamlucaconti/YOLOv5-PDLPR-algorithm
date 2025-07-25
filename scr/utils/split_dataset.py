import os, shutil, random
from glob import glob

# ------------ CONFIG -------------------------------------------------
dataset_root = os.getcwd()
subset_root  = os.path.join(dataset_root, "ccpd_test")
max_test     = 1000
random_seed  = 42
# --------------------------------------------------------------------

# (1) reset cartella destinazione
if os.path.exists(subset_root):
    print(f" La cartella '{subset_root}' esiste già. La elimino.")
    shutil.rmtree(subset_root)
os.makedirs(subset_root, exist_ok=True)

# (2) mapping split
split_files = {
    "blur":      os.path.join(dataset_root, "splits/ccpd_blur.txt"),
    "challenge": os.path.join(dataset_root, "splits/ccpd_challenge.txt"),
    "db":        os.path.join(dataset_root, "splits/ccpd_db.txt"),
    "rotate":    os.path.join(dataset_root, "splits/ccpd_rotate.txt"),
    "tilt":      os.path.join(dataset_root, "splits/ccpd_tilt.txt"),
    "fn":        os.path.join(dataset_root, "splits/ccpd_fn.txt"),
    "weather":   os.path.join(dataset_root, "ccpd_weather"),
    "base":      os.path.join(dataset_root, "ccpd_base"),
}

# --------- UTILITY ---------------------------------------------------
def read_paths_from_txt(txt):
    with open(txt) as f:
        return [ln.strip() for ln in f if ln.strip()]

def list_images(dir_path):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(dir_path, "**", e), recursive=True))
    return [os.path.relpath(p, dataset_root) for p in files]

def copy_images(rel_paths, subset_name, limit=None):
    out_dir = os.path.join(subset_root, subset_name)
    os.makedirs(out_dir, exist_ok=True)
    rel_paths = rel_paths[:limit] if limit else rel_paths
    for rp in rel_paths:
        src = os.path.join(dataset_root, rp)
        dst = os.path.join(out_dir, os.path.basename(rp))
        if os.path.exists(src):
            shutil.copy2(src, dst)

# ---------  costruisci set immagini train ---------------------------
train_dir = os.path.join(dataset_root, "ccpd_subset_base", "train")
train_imgs = set(os.path.basename(p)
                 for p in glob(os.path.join(train_dir, "**", "*.jpg"),
                               recursive=True))

print(f"  Trovate {len(train_imgs)} immagini nel training set da evitare")

# ------------------- MAIN LOOP --------------------------------------
random.seed(random_seed)

for split_name, path in split_files.items():

    # ottieni lista immagini per lo split
    if os.path.isfile(path):
        rel_paths = read_paths_from_txt(path)
    elif os.path.isdir(path):
        rel_paths = list_images(path)
    else:
        print(f"  Percorso non trovato per '{split_name}': {path}")
        continue

    # filtra se è lo split base
    if split_name == "base":
        before = len(rel_paths)
        rel_paths = [p for p in rel_paths
                     if os.path.basename(p) not in train_imgs]
        print(f"     Filtrate {before-len(rel_paths)} immagini già nel train")

    random.shuffle(rel_paths)
    print(f"  Copio max {max_test} immagini di '{split_name}' …")
    copy_images(rel_paths, split_name, max_test)

print(" Subset creato in:", subset_root)