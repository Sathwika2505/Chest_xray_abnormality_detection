import os
import zipfile
import pandas as pd
import boto3
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import io
import random
import shutil
import pathlib
from data_extraction import read_csv_from_s3

def download_and_extract_zip_from_s3(bucket_name, zip_file_key, local_dir):
    s3 = boto3.client('s3')
    with BytesIO() as zip_buffer:
        s3.download_fileobj(bucket_name, zip_file_key, zip_buffer)
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
            file_count = len([name for name in zip_ref.namelist() if name.endswith('.jpg')])
    print(f"Files extracted to {local_dir}")
    print(f"Total number of images in zip file: {file_count}")
    return file_count

def organize_images_by_class(csv_df, images_dir, output_dir):
    if csv_df is None:
        print("CSV DataFrame is None. Exiting the function.")
        return
    
    saved_files = []
    missing_files = []
    
    # Iterate over each row in the DataFrame
    for index, row in csv_df.iterrows():
        image_id = row['image_id']
        class_name = row['class_name']
        print("class_name-----------",class_name)
        # Check if the image file exists in the images_dir
        image_file = f"{image_id}.jpg"
        image_path = os.path.join(images_dir, "train", image_file)  # Adjusted path
        
        if os.path.exists(image_path):
            # Create the label directory if it doesn't exist
            if class_name == "Nodule":
                label_dir = os.path.join(output_dir, class_name, "Mass")  # Using "Nodule/Mass" path
            else:
                label_dir = os.path.join(output_dir, class_name)
                
            pathlib.Path(label_dir).mkdir(parents=True, exist_ok=True)
            
            # Destination path for the image
            dest_path = os.path.join(label_dir, image_file)
            
            # Copy the image to the respective label directory
            shutil.copy(image_path, dest_path)
            
            saved_files.append(dest_path)
        else:
            print(f"Image {image_file} not found in {images_dir}.")
            missing_files.append(image_file)
    
    print("Images saved successfully.")
    print(f"Total images missing: {len(missing_files)}")
    return saved_files, missing_files

def open_random_image(path):
    try:
        all_files = os.listdir(path)
        random_image_file = random.choice(all_files)
        image_path = os.path.join(path, random_image_file)
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error opening image from {path}: {e}")
        return None

def save_random_images_from_each_class(base_dir):
    class_dirs = {
        "Aortic enlargement": "Aorticenlargement.jpg",
        "Atelectasis": "Atelectasis.jpg",
        "Calcification": "Calcification.jpg",
        "Cardiomegaly": "Cardiomegaly.jpg",
        "Consolidation": "Consolidation.jpg",
        "ILD": "ILD.jpg",
        "Infiltration": "Infiltration.jpg",
        "Lung Opacity": "LungOpacity.jpg",
        "No finding": "Nofinding.jpg",
        "Nodule": "Nodule.jpg",
        "Other lesion": "Otherlesion.jpg",
        "Pleural effusion": "Pleuraleffusion.jpg",
        "Pleural thickening": "Pleuralthickening.jpg",
        "Pneumothorax": "Pneumothorax.jpg",
        "Pulmonary fibrosis": "Pulmonaryfibrosis.jpg"
    }

    for class_name, filename in class_dirs.items():
        class_dir = os.path.join(base_dir, class_name)
        if class_name == "Nodule":
            # For "Nodule" class, iterate through the subdirectories
            class_subdirs = [subdir for subdir in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, subdir))]
            if class_subdirs:
                random_subdir = random.choice(class_subdirs)
                random_image = open_random_image(os.path.join(class_dir, random_subdir))
            else:
                print(f"No subdirectories found for class {class_name}")
                continue
        else:
            random_image = open_random_image(class_dir)

        if random_image:
            random_image.save(filename)
        else:
            print(f"No image saved for class {class_name}")

def main():
    # Define your parameters here
    bucket_name = 'deeplearning-mlops-demo'
    zip_file_key = 'trainimages.zip'
    csv_file_key = 'train.csv'
    local_dir = 'local_extracted'
    output_dir = 'organized_images'

    # Step 1: Download and extract zip file
    zip_file_count = download_and_extract_zip_from_s3(bucket_name, zip_file_key, local_dir)

    # Step 2: Read the CSV file
    csv_df = read_csv_from_s3(bucket_name, csv_file_key)

    # Step 3: Organize images by class
    saved_files, missing_files = organize_images_by_class(csv_df, local_dir, output_dir)

    # Step 4: Save a random image from each class
    save_random_images_from_each_class(output_dir)

if __name__ == "__main__":
    main()
