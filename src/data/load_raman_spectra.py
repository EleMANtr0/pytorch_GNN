import numpy as np
from glob import glob
import json
import warnings
import re

from ..utils import qxrd_apc as apc

def load_raman_data(
        model_wavenumber_values,
        raman_data_directory_path='data/raw/raman/',
        wavelength=None,
        verbose=False,
        zero_pad=True):

    if wavelength:
        file_paths_list = glob(raman_data_directory_path+f'*/*__{wavelength}__*.txt')
    else:
        file_paths_list = glob(raman_data_directory_path+'*/*.txt')
    mineral_names = []
    raman_spectra = []
    wavelengths = []
    orientation_vecs = []
    raman_ids = []
    for fp in file_paths_list:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try:
                mineral_name, rruff_id, wavelength, raman_spectrum, orientation_vec = load_single_raman_spectrum(model_wavenumber_values,fp,zero_pad)
                mineral_names.append(mineral_name)
                raman_spectra.append(raman_spectrum)
                orientation_vecs.append(orientation_vec)
                wavelengths.append(wavelength)
                raman_ids.append(rruff_id)
            except RuntimeWarning:
                print("runtime warning at ", fp)
                continue
            except Exception as e:
                if verbose:
                    print(e)
                    print(f"problem file: {fp}")
                    print("")

    return file_paths_list, raman_ids, mineral_names, raman_spectra, wavelengths, orientation_vecs

pattern = re.compile(r"##ORIENTATION.*?[\(\[]\s*([-\d\s\.]+)\s*[\)\]].*?[\(\[]\s*([-\d\s\.]+)\s*[\)\]]")

def load_single_raman_spectrum(
        model_wavenumber_values,
        file_path,
        zero_pad=True):
    mineral_name = file_path.split('\\')[-1].split('/')[-1].split('__')[0]
    
    raw_id_part = file_path.split('__')[-1]
    rruff_id = raw_id_part.split('-')[0].replace('.txt', '')
    
    wavelength = int(file_path.split('__Raman__')[-1].split('__')[0])
    if "_oriented" in file_path:
        orientation = 1
    elif "_unoriented" in file_path:
        orientation = 0
    orientation_vec = [0] * 7
    if orientation:
        with open(file_path, "r") as f:
            contents = f.read(3000)
            if "##ORIENTATION" not in contents:
                pass
            else:
                raw = pattern.search(contents).groups()
                orientation_vec = [1] + [int(i) for row in raw for i in row.split(" ")]

    temp_apc = apc.TopLevel(file_path,twotheta_ranges=[(0.0,100000.0)],print_warnings=False)
    if 0 in temp_apc.input_profile.xy_data[1]:
        raise Exception('Model wavenumbers too broad for this spectrum. Skip.')
    raman_spectrum = process_raman_spectrum(temp_apc.input_profile.xy_data,model_wavenumber_values,zero_pad)
    return mineral_name, rruff_id, wavelength, raman_spectrum, orientation_vec


def process_raman_spectrum(xy,model_twotheta_values,zero_pad=True):
    if zero_pad:
        intensity_interpolated = np.interp(model_twotheta_values,xy[0],xy[1],left=0.0,right=0.0)
    else:
        intensity_interpolated = np.interp(model_twotheta_values,xy[0],xy[1])
    intensity_normalized = np.multiply(intensity_interpolated,1.0/np.max(intensity_interpolated))
    return intensity_normalized


def load_ir_data(ir_dir, ir_wavenumbers):
    ir_dict_id = {}
    ir_dict_name = {}
    for fp in glob(ir_dir + '*/*.txt'):
        try:
            filename = fp.split('\\')[-1].split('/')[-1]
            mineral_name = filename.split('__')[0]
            
            if '__' in filename:
                base_id = filename.split('__')[-1].split('-')[0].replace('.txt', '')
            else:
                base_id = filename.split('-')[0].replace('.txt', '')
                
            if not base_id.startswith('R'):
                base_id = 'R' + base_id
            
            temp_apc = apc.TopLevel(fp, twotheta_ranges=[(0.0, 100000.0)], print_warnings=False)
            ir_spectrum = process_raman_spectrum(temp_apc.input_profile.xy_data, ir_wavenumbers, True)
            
            if base_id:
                ir_dict_id[base_id] = ir_spectrum
            ir_dict_name[mineral_name] = ir_spectrum
        except:
            pass
    return ir_dict_id, ir_dict_name
