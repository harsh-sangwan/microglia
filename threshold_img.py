import os
#import cv2
import numpy as np
import nibabel as nib


def write_nifti(path, volume):
    '''
    writeNifti(path,volume)

    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path.
    '''
    if (path.find('.nii') == -1):
        path = path + '.nii.gz'
    # Save volume with adjusted orientation
    # --> Swap X and Y axis to go from (y,x,z) to (x,y,z)
    # --> Show in RAI orientation (x: right-to-left, y: anterior-to-posterior, z: inferior-to-superior)
    affmat = np.eye(4)
    affmat[0, 0] = affmat[1, 1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume, 0, 1), affine=affmat)
    nib.save(NiftiObject, os.path.normpath(path))


def read_nifti(path):
    '''
    volume = readNifti(path)

    Reads in the NiftiObject saved under path and returns a Numpy volume.
    '''
    if (path.find('.nii') == -1):
        path = path + '.nii'
    NiftiObject = nib.load(path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj, 0, 1)
    return volume


if __name__ == "__main__":
    input_file = 'raw/patchvolume_1874.nii.gz'
    output_file = 'output/patchvolume_1874_thresh.nii.gz'
    threshold = 0.2     #must be between 0 and 1

    vol = read_nifti(input_file)

    #normalize values to be between 0 and 1
    vol /= np.max(vol)

    #change values above threshold to 1 else zero
    vol[vol  >= threshold] = 1
    vol[vol < threshold] = 0

    write_nifti(output_file, vol)
