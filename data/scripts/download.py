import os
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi

def setup_kaggle_credentials():

    kaggle_dir = os.path.expanduser('~/.kaggle')
    
    # Check if credentials exist
    if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
        print("Kaggle API credentials not found.")
        print("Please download your kaggle.json from:")
        print("https://www.kaggle.com/account")
        print(f"And place it in: {kaggle_dir}")
        return False
    
    # Ensure proper permissions
    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
    return True

def download_dataset(dataset_name, path='.', unzip=True):

    try:
        print(f"Downloading dataset: {dataset_name}")
        
        api = KaggleApi()
        api.authenticate()
        
        os.makedirs(path, exist_ok=True)
        
        api.dataset_download_files(
            dataset=dataset_name,
            path=path,
            unzip=unzip
        )
        
        print(f"Dataset downloaded successfully to {path}")
        print("\nDownloaded files:")
        
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                print(f" - {file}")
                
    except Exception as e:
        print(f"Error downloading dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download datasets from Kaggle')
    
    parser.add_argument('dataset', type=str, 
                        help='Dataset identifier in format owner/dataset-name')
    
    parser.add_argument('--path', '-p', type=str, default='./data',
                        help='Directory to download files to (default: ./data)')
    
    parser.add_argument('--no-unzip', action='store_true',
                        help='Do not unzip the downloaded files')
    
    args = parser.parse_args()
    
    if setup_kaggle_credentials():
        download_dataset(
            dataset_name=args.dataset,
            path=args.path,
            unzip=not args.no_unzip
        )

if __name__ == "__main__":
    main()