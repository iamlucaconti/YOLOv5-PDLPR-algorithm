import tarfile
import os

def make_tar_fast(source_dir, output_filename):
    with tarfile.open(output_filename, "w") as tar:  # "w" = tar senza compressione
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print(f" Archivio .tar creato: {output_filename}")

dataset_root = os.getcwd()
folder_to_compress = os.path.join(dataset_root, "ccpd_test")
output_tar = os.path.join(dataset_root, "ccpd_test.tar")  # .tar, NON .tar.xz

make_tar_fast(folder_to_compress, output_tar)
