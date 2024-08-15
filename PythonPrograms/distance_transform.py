import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, label

def calculate_biliary_duct_segment_volumes(biliary_duct_file, skeletonized_file):
    biliary_img = nib.load(biliary_duct_file)
    skeleton_img = nib.load(skeletonized_file)
    biliary_data = biliary_img.get_fdata()
    skeleton_data = skeleton_img.get_fdata()
    
    labeled_components, num_components = label(biliary_data > 0)
    
    # distance transfrom
    distance_transform = distance_transform_edt(skeleton_data)
    
    segment_volumes = {}
    
    total_volume = 0
    voxel_volume = np.prod(biliary_img.header.get_zooms())  # voxel dimensions from header
    
    for i in range(1, num_components + 1):
        component_mask = (labeled_components == i)
        component_volume = distance_transform[component_mask].sum()
        
        component_volume_in_mm3 = component_volume * voxel_volume
        segment_volumes[f"Segment_{i}"] = component_volume_in_mm3
        
        total_volume += component_volume_in_mm3
    
    # calculate drain percents
    segment_percentages = {
        segment: (volume / total_volume) * 100
        for segment, volume in segment_volumes.items()
    }
    
    return total_volume, segment_volumes, segment_percentages

biliary_duct_file = 'path_to_biliary_duct_segmentation.nii'
skeletonized_file = 'path_to_skeletonized_segmentation.nii'
total_volume, segment_volumes, segment_percentages = calculate_biliary_duct_segment_volumes(biliary_duct_file, skeletonized_file)

print(f"Total Biliary Duct Volume: {total_volume} mm^3\n")
for segment, volume in segment_volumes.items():
    print(f"{segment}: {volume} mm^3 ({segment_percentages[segment]:.2f}%)")
