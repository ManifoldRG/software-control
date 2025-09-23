import os
import subprocess

from huggingface_hub import hf_hub_download, list_repo_files


def download_agentnet_from_hf():
    repo_id = "xlangai/AgentNet"
    folder = "ubuntu_images"
    local_dir = "./external_data/agentnet/ubuntu_images"

    all_files = list_repo_files(repo_id, repo_type="dataset")
    folder_files = [f for f in all_files if f.startswith(folder + "/")]

    os.makedirs(local_dir, exist_ok=True)
    for file in folder_files:
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file, local_dir=local_dir)

    hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=folder + "", local_dir=local_dir)

    subprocess.run(["zip", "-s", "0", local_dir + ".zip", "--out", local_dir + "-full.zip"])
    subprocess.run(["unzip", local_dir + "-full.zip", "-d", local_dir])


def download_osworld_verified_traj_from_hf(hf_file_name: str):
    repo_id = "xlangai/ubuntu_osworld_verified_trajs"
    local_dir = f"./external_data/osworld_verified/{hf_file_name}"

    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=hf_file_name + ".zip", local_dir=local_dir)


if __name__ == "__main__":
    # download_agentnet_from_hf()
    download_osworld_verified_traj_from_hf("UI-TARS-0717-100step")
