# rm & mv

### move with exclusion

```bash
# move all files except dir into dir, a usual trick.
mv !(dir) dir 

# exclude more
mv !(dir1|dir2) dir1
```


### remove with exclusion

```bash
# rm all files except `file`
rm !(file)

rm !(file1|file2)

# prompt(i) and verbose(v)
rm -vi *
```


