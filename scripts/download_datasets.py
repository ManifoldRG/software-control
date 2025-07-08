import yaml
from pathlib import Path
from huggingface_hub import snapshot_download

class DatasetDownloader:
    def __init__(self, config_path="scripts/dataset_registry.yaml", data_root="external_data"):
        with open(config_path, "r") as f:
            self.registry = yaml.safe_load(f)
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)

    def download(self, dataset_name: str):
        try:
            cfg = self.registry[dataset_name]
        except KeyError:
            raise ValueError(f"Dataset {dataset_name} not found in registry")
        
        repo_id = cfg["repo_id"]
        dest = self.data_root / dataset_name
        dest.mkdir(exist_ok=True)
        
        print(f"Downloading {repo_id} from Hugging Face to {dest}")
        snapshot_download(repo_id=repo_id,
                          repo_type="dataset", 
                          local_dir=dest,
                          local_dir_use_symlinks=False)

    def download_all(self):
        for name in self.registry:
            self.download(name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to download (or 'all')")
    args = parser.parse_args()
    downloader = DatasetDownloader()
    if args.dataset == "all" or args.dataset is None:
        downloader.download_all()
    else:
        downloader.download(args.dataset)