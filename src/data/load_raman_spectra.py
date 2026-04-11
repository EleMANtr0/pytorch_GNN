import numpy as np
from glob import glob
import warnings
import re

from utils import qxrd_apc as apc


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
    max_int = []
    for fp in file_paths_list:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try:
                mineral_name, rruff_id, wavelength, raman_spectrum, max_intensity, orientation_vec = load_single_raman_spectrum(model_wavenumber_values, fp, zero_pad)
                mineral_names.append(mineral_name)
                raman_spectra.append(raman_spectrum)
                max_int.append(max_intensity)
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

    return file_paths_list, raman_ids, mineral_names, raman_spectra, wavelengths, orientation_vecs, max_int

pattern = re.compile(r"##ORIENTATION.*?[\(\[]\s*([-\d\s\.]+)\s*[\)\]].*?[\(\[]\s*([-\d\s\.]+)\s*[\)\]]")

def load_single_raman_spectrum(
        model_wavenumber_values,
        file_path,
        zero_pad=True):
    mineral_name = file_path.split('\\')[-1].split('/')[-1].split('__')[0]
    
    raw_id_part = file_path.split('__')[1]
    rruff_id = raw_id_part.split('-')[0]
    
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
    raman_spectrum, max_intensity = process_raman_spectrum(temp_apc.input_profile.xy_data,model_wavenumber_values,zero_pad)
    return mineral_name, rruff_id, wavelength, raman_spectrum, max_intensity, orientation_vec


def process_raman_spectrum(xy,model_twotheta_values,zero_pad=True):
    if zero_pad:
        intensity_interpolated = np.interp(model_twotheta_values,xy[0],xy[1],left=0.0,right=0.0)
    else:
        intensity_interpolated = np.interp(model_twotheta_values,xy[0],xy[1])
    intensity_normalized = intensity_interpolated / np.max(intensity_interpolated)
    return intensity_normalized, np.max(intensity_interpolated)


def load_ir_data(ir_dir, ir_wavenumbers, zero_pad):
    ir_dict_id = {}
    ir_dict_name = {}
    for fp in glob(ir_dir + '*.txt'):
        try:
            filename = fp.split('\\')[-1].split('/')[-1]
            mineral_name = filename.split('__')[0]
            
            
            raw_id_part = filename.split('__')[1]
            rruff_id = raw_id_part.split('-')[0]
            
            temp_apc = apc.TopLevel(fp, twotheta_ranges=[(0.0, 100000.0)], print_warnings=False)
            ir_spectrum, max_ir = process_raman_spectrum(temp_apc.input_profile.xy_data, ir_wavenumbers, zero_pad=zero_pad)
            
            ir_dict_id[rruff_id] = (ir_spectrum, max_ir)
            ir_dict_name[mineral_name] = (ir_spectrum, max_ir)
        except:
            pass
    return ir_dict_id, ir_dict_name
