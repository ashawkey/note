## s5cmd

A better S3 filesystem manager.

### Install
Download binary from [releases](https://github.com/peak/s5cmd/releases) and add it to PATH.

### Setup
Directly use ENV is the simplest:
```bash
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
export S3_ENDPOINT_URL=https://xxx
```

Also can use `~/.aws/credentials`:
```
[$PROFILE_NAME_YOU_WANT]
region=us-east-1
aws_access_key_id=xxx
aws_secret_access_key=xxx
s3=
 endpoint_url=https://xxx
s3api=
 endpoint_url=https://xxx
 payload_signing_enabled=true
```

### Usage
```bash
# ls all buckets
s5cmd ls

# download to local
s5cmd cp s3://bucket/file .

# upload
s5cmd cp file s3://bucket/
s5cmd cp dir s3://bucket/dir # no -r needed

# delete
s5cmd rm s3://bucket/file

# exclude and include
s5cmd rm --exclude "*.log" --exclude "*.txt" 's3://bucket/logs/2020/*'
s5cmd cp --include "*.log" 's3://bucket/logs/2020/*' .

# du
s5cmd du --humanize s3://bucket/
```
