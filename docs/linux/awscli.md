## AWS

```bash
pip install awscli
```


### Setup

First configure it using your AK & SK:

```bash
aws configure
```

It will create credentials under `~/.aws/credentials`

```config
[default]
aws_access_key_id=<default access key>
aws_secret_access_key=<default secret key>
```

To use a different endpoint than amazon, config an alias in your `.bashrc`:

```bash
alias aws="aws --endpoint-url=http://<ip>:<port>"
```


### Usage

```bash
# list S3 buckets under your access key
aws s3 ls

# debug (e.g., to check your endpoint ip)
aws s3 ls --debug

# create a bucket
aws s3 mb s3://myBucket

# delete a bucket (!)
aws s3 rb s3://myBucket

# list files in a bucket
aws s3 ls s3://myBucket

# upload local file to a bucket
aws s3 cp local_file s3://myBucket/remote_file

# download remote file from bucket
aws s3 cp s3://myBucket/remote_file local_file
aws s3 cp --recursive s3://myBucket/remote_folder local_folder

# delete remote file
aws s3 rm s3://myBucket/remote_file
```


### Megfile

A management tool for both S3 and normal dataset.

```bash
pip install megfile

# config s3 access key
megfile config s3 AK SK --addressing-style virtual --endpoint-url http://<ip>:<port>
```

#### CLI

```bash
megfile ls s3://bucket
megfile cat s3://bucket/file
megfile cp s3://bucket/file local_file
```

#### API

```python
from megfile import smart_open, smart_glob
from megfile.smart_path import SmartPath

files = smart_glob('s3://bucket/*.jpg')
for file in files:
    with smart_open(file, 'r') as fp:
		content = fp.read()
```

