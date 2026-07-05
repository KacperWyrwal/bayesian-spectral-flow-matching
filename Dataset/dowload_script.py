from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/timewarp",
    repo_type="dataset",
    allow_patterns="AD-3/*",
    local_dir="./Dataset/AD3"
)