## huggingface cli

```bash
# install
pip install huggingface_hub

# login (use https://huggingface.co/settings/tokens)
huggingface-cli login
```

### upload models to a repo

* Create the repo from website first.

* upload by CLI:

  ```bash
  huggingface-cli upload <user/repo> <local path> <remote path>
  # example for uploading everything to remote root dir
  huggingface-cli upload <user/repo> . .
  ```

  