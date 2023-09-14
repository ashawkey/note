## Huggingface API



```bash
pip install -U huggingface_hub

# login with token in https://huggingface.co/settings/tokens (note the permission of read/write)
huggingface-cli login
```



### Upload model to repo

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



