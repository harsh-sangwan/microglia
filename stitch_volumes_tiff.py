from psutil import virtual_memory
import numpy as np
import os
import shutil
import datetime
import functools
import multiprocessing
from libtiff import TIFF, TIFFfile, TIFFimage
import cv2
import numpy as np
from util_pkg import filehandling
from util_pkg import plotting
import pickle
import multiprocessing as mp



path_media          = '/media/harshsangwan/51A631B1183117DF/Microglia/'
dataset             = "cx3cr1gfp_145/Patches/"
path_project        = os.path.join(path_media, dataset)
path_data           = os.path.join(path_project, "Sync")
path_segmentation   = os.path.join(path_project, "somata_pred")
path_pd             = os.path.join(path_segmentation,  "patchvolume_{}.nii.gz") 
path_tiff_out       = os.path.join(path_project, "tiffs")
path_out            = os.path.join(path_tiff_out, "z_{}.tif")

if not os.path.isdir(path_tiff_out):
    os.mkdir(path_tiff_out)

# Load data
print("path data:" + str(path_data))
region_path = os.path.join(path_data, 'region')
region = filehandling.pload(region_path)
print(plotting.print_dict(region))
print(plotting.print_dict(region["patches"][0]))
print(region["partitioning"]["patches_per_dim"] * region["partitioning"]["patch_size"])

patch_size = region["partitioning"]["patch_size"] # Y X Z
patch_overlap = region["partitioning"]["patch_overlap"]
patches_per_dim = region["partitioning"]["patches_per_dim"] # Y X Z
cropping_offset = region["partitioning"]["cropping_offset"] # Y X Z
cropping_boundingbox = region["partitioning"]["cropping_boundingbox"] # Y X Z

original_size = region["dataset"]["bounding_box"] # X Y Z

mem = virtual_memory()
z_layer_space = int(patches_per_dim[0]) * \
    int(patches_per_dim[1]) * (patch_size[0]**3)

max_mem = 0.3 * mem.available
max_patches = int(max_mem / patch_size[0] ** 3) + 1

print("Available memory : {} GB".format(mem.available / (1000**3)))
print("All patches for one Z Layer need {} GB".format(z_layer_space / (1000**3)))
print("Cutting {} patches at once".format(max_patches))
print(f"PPD {patches_per_dim}")
print({len(region["patches"])})
print(region["patches"][3000])

def cut_z_range(start, end):
    for z in range(start, end):
    # for z in range(0, patches_per_dim[2]):
        # List of all patches in a z-step
        z_list = []
        for x in range(patches_per_dim[1]):
            for y in range(patches_per_dim[0]):
                pt  = [patch for patch in region["patches"] if patch["patchstep"] == [y, x, z]]
                p   = [patch for patch in pt if not patch["locationgroup"] == "Outside"] 
                lg  = "Outside"
                if len(p) > 0:
                    lg = p[0]["locationgroup"]
                    print("\t" * 10*start + f"{z} {x} {y} {len(p)} {lg}", end="\r", flush=True)
                try:
                    if len(p) > 0:
                        z_list.append(p[0]) 
                except Exception as ex:
                    print("{} for {}".format(ex, [y, x, z]))
        print("\nDone searching patches")
        for z_layer in range(patch_size[2]):
            if os.path.exists(path_out.format(z*patch_size[2] + z_layer)):
                print(path_out.format(z*patch_size[2] + z_layer) + " already exists, skipping...")
            else:
                # Iterate over every entry in z-direction
                # Blank image
                z_image = np.zeros((original_size[0], original_size[1]))
                for en, item in enumerate(z_list):
                    if item["locationgroup"] == "Core" or item["locationgroup"] == "Boundary":
                        #TODO Offset des ganzen Volumens berücksichtigen
                        lc = ""
                        if item["locationgroup"] == "Core":
                            lc = "C"
                        else:
                            lc = "B"
                        print("\t"*10*start + "Reading {} \t({} of {}) - \t{}\t\t - {} of {}|".format(item["id"], en, len(z_list), lc, z, patches_per_dim[2]),end="\r",flush=True)
                        volume = filehandling.read_nifti(path_pd.format(item["id"]))
                        i_off = item["offset"]
                        z_image[i_off[0]: i_off[0] + patch_size[0] - patch_overlap, i_off[1]:i_off[1] + patch_size[1] - patch_overlap] = volume[:-patch_overlap,:-patch_overlap, z_layer]
                z_image = z_image.astype(np.int8)
                if np.amax(z_image) == 0:
                    print("Omiting black image {}".format(z*patch_size[0] + z_layer))
                else:
                    print("\t"*10*start + "Writing {} of {}".format(
                        z*patch_size[0] + z_layer, patch_size[0] * (patches_per_dim[2] - 1) + patch_size[2] - 1))
                    tiff = TIFFimage(z_image, description='')
                    tiff.write_file(path_out.format(z*patch_size[0] + z_layer),
                                    compression='lzw', verbose=False)

# cut_z_range(0, patches_per_dim[2])

pool = mp.Pool(mp.cpu_count() - 5)
beg = datetime.datetime.now()

for i in range(1, patches_per_dim[2]):
    pool.apply_async(cut_z_range, args=(i-1,i))

pool.close()
pool.join()

