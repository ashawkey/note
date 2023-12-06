## Huggingface

### Install

```bash
# install
pip install huggingface_hub

# login (use https://huggingface.co/settings/tokens)
huggingface-cli login
```

### upload models to a repo

using python API:

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

using CLI:

* Create the repo from website first.

* upload by

  ```bash
  huggingface-cli upload <user/repo> <local path> <remote path>
  # example for uploading everything to remote root dir
  huggingface-cli upload <user/repo> . .
  ```

  

### Downloads a repo

```bash
huggingface-cli download <user/repo>
```

