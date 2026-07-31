```bash
### diff two folder 
# -b: ignore space
# -u: output 3 lines of unified context
# -r: recursive
diff -bur folder1/ folder2 | filterdiff -i '*.py'
diff -bur folder1/ folder2 | filterdiff -e '*.log'
```

