## Huggingface

### Install

```bash
# install
pip install huggingface_hub

# login (use https://huggingface.co/settings/tokens)
huggingface-cli login
```


### upload models to a repo

##### using python API:

```python
from huggingface_hub import HfApi
api = HfApi()

### create repo
api.create_repo(repo_id="repo", private=True)

### upload file
api.upload_file(
    path_or_fileobj="/path/to/obj",
    path_in_repo="obj",
    repo_id="user/repo",
)

### upload folder (to root dir)
api.upload_folder(
    folder_path="/path/to/dir",
    repo_id="user/repo",
    repo_type="model", # dataset, space
)
```

##### using CLI:

If the repo is not existing, it will be created automatically!

```bash
# upload local files to remote repo
huggingface-cli upload <user/repo> <local path> <remote path>

# upload everything to remote root dir
huggingface-cli upload <user/repo> . .

# upload a single file 
huggingface-cli upload <user/repo> ./path/to/myfile # remote default to .
huggingface-cli upload <user/repo> ./path/to/myfile /path/to/remote

# upload multiple files: use  --include --exclude
huggingface-cli upload <user/repo> --include="*.mp4" --exclude="unwanted*"
```


### download models

##### using CLI:

By default these commands will download to `~/.cache/huggingface/hub`, use `--local-dir` to change it!

```bash
# download single/multiple files to current dir
huggingface-cli download <user/repo> <file1> [<file2> ...] --local-dir .

huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --exclude "*.fp16.*"*

# download entire repo
huggingface-cli download <user/repo> --local-dir .
```

