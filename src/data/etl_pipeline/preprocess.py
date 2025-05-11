import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.dcmread(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def process_file(args):
    dicom_path, output_dir = args
    try:
        img_data = read_xray(dicom_path)
        img = Image.fromarray(img_data)

        base = os.path.splitext(os.path.basename(dicom_path))[0]
        output_path = os.path.join(output_dir, base + '.png')
        img.save(output_path)
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def main(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    dicom_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith('.dicom') or f.endswith('.dcm')
    ]
    args_list = [(f, output_directory) for f in dicom_files]
    
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, args_list), total=len(args_list), desc="Processing DICOM images"))

if __name__ == "__main__":
    train_input_directory = 'train'
    train_output_directory = 'data/train'
    main(train_input_directory, train_output_directory)

    test_input_directory = 'test'
    test_output_directory = 'data/test'
    main(test_input_directory, test_output_directory)
