from radiometric_normalization.wrappers import pif_wrapper
from radiometric_normalization.wrappers import transformation_wrapper
from radiometric_normalization.wrappers import normalize_wrapper
from radiometric_normalization import gimage
from radiometric_normalization import pif

## OPTIONAL
import logging
import numpy
import subprocess
from osgeo import gdal
from radiometric_normalization.wrappers import display_wrapper

logging.basicConfig(level=logging.DEBUG)
##

## OPTIONAL - Cut dataset to colocated sub scenes and create and BGRN image
# LC08_L1TP_044034_20170105_20170218_01_T1 is the older scene and so it is set as the reference.

band_mapping = [{'name': 'blue', 'L8': 'B2'}, {'name': 'green', 'L8':'B3'}, {'name': 'red', 'L8': 'B4'}, {'name': 'nir', 'L8': 'B5'}]

full_candidate_basename = 'D:\TOULOUSE2\MONTUS\LC08_01_044_034_LC08_L1TP_044034_20170427_20170515_01_T1'
full_reference_basename = 'D:\TOULOUSE2\MONTUS\LC08_01_044_034_LC08_L1TP_044034_20170105_20170515_01_T1'
candidate_basename = 'candidate'
reference_basename = 'reference'
full_candidate_filenames = ['{}_{}.TIF'.format(full_candidate_basename, b['L8']) for b in band_mapping]
candidate_filenames = ['{}_{}.TIF'.format(candidate_basename, b['name']) for b in band_mapping]
full_reference_filenames = ['{}_{}.TIF'.format(full_reference_basename, b['L8']) for b in band_mapping]
reference_filenames = ['{}_{}.TIF'.format(reference_basename, b['name']) for b in band_mapping]

for full_filename, cropped_filename in zip(full_candidate_filenames, candidate_filenames):
    subprocess.check_call(['gdal_translate', '-projwin', '545000', '4136000', '601000', '4084000', full_filename, cropped_filename])

for full_filename, cropped_filename in zip(full_reference_filenames, reference_filenames):
    subprocess.check_call(['gdal_translate', '-projwin', '545000', '4136000', '601000', '4084000', full_filename, cropped_filename])

band_gimgs = {}
for cropped_filename in candidate_filenames:
    band = cropped_filename.split('_')[1].split('.TIF')[0]
    band_gimgs[band] = gimage.load(cropped_filename)

candidate_path = 'D:\TOULOUSE2\MONTUS\python_module\candidate.tif'
combined_alpha = numpy.logical_and.reduce([b.alpha for b in band_gimgs.values()])
temporary_gimg = gimage.GImage([band_gimgs[b].bands[0] for b in ['blue', 'green', 'red', 'nir']], combined_alpha, band_gimgs['blue'].metadata)
gimage.save(temporary_gimg, candidate_path)

band_gimgs = {}
for cropped_filename in reference_filenames:
    band = cropped_filename.split('_')[1].split('.TIF')[0]
    band_gimgs[band] = gimage.load(cropped_filename)

reference_path = 'D:\TOULOUSE2\MONTUS\python_module\reference.tif'
combined_alpha = numpy.logical_and.reduce([b.alpha for b in band_gimgs.values()])
temporary_gimg = gimage.GImage([band_gimgs[b].bands[0] for b in ['blue', 'green', 'red', 'nir']], combined_alpha, band_gimgs['blue'].metadata)
gimage.save(temporary_gimg, reference_path)
##

parameters = pif.pca_options(threshold=100)
pif_mask = pif_wrapper.generate(candidate_path, reference_path, method='filter_PCA', last_band_alpha=True, method_options=parameters)

## OPTIONAL - Save out the PIF mask
candidate_ds = gdal.Open(candidate_path)
metadata = gimage.read_metadata(candidate_ds)
pif_gimg = gimage.GImage([pif_mask], numpy.ones(pif_mask.shape, dtype=numpy.bool), metadata)
gimage.save(pif_gimg, 'D:\TOULOUSE2\MONTUS\python_module\PIF_pixels.tif')
##

transformations = transformation_wrapper.generate(candidate_path, reference_path, pif_mask, method='linear_relationship', last_band_alpha=True)

## OPTIONAL - View the transformations
print (transformations)
##

normalised_gimg = normalize_wrapper.generate(candidate_path, transformations, last_band_alpha=True)
result_path = 'D:\TOULOUSE2\MONTUS\python_module\normalized.tif'
gimage.save(normalised_gimg, result_path)

## OPTIONAL - View the effect on the pixels (SLOW)
from radiometric_normalization.wrappers import display_wrapper
display_wrapper.create_pixel_plots(candidate_path, reference_path, 'Original', limits=[0, 30000], last_band_alpha=True)
display_wrapper.create_pixel_plots(result_path, reference_path, 'Transformed', limits=[0, 30000], last_band_alpha=True)
display_wrapper.create_all_bands_histograms(candidate_path, reference_path, 'Original', x_limits=[4000, 25000], last_band_alpha=True)
display_wrapper.create_all_bands_histograms(result_path, reference_path, 'Transformed', x_limits=[4000, 25000], last_band_alpha=True)
##
