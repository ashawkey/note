import os
import re
import glob

files = glob.glob('docs/**/*.md', recursive=True)

# fix math blocks to be correctly rendered by mathjax:
# 1. $$..$$ will always be preceded and followed by an empty line, and there is no blank line inside.
# 2. always wrap all content within \displaylines{}

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()

    # naive matching $$...$$
    outside = True
    modified = False
    for i, line in enumerate(lines):
        if re.match('^\$\$', line) is not None:
            if outside:
                if lines[i+1] != '\displaylines{\n':
                    lines[i] = '\n$$\n\displaylines{\n'
                    modified = True
                outside = False
            else:
                if lines[i-1] != '}\n':
                    lines[i] = '}\n$$\n\n'
                    modified = True
                outside = True

        # remove blank lines inside
        if not outside and re.match('^\s*\n$', line):
            lines[i] = ''
            modified = True
            
    # remove excessive blank lines
    cnt = 0
    for i, line in enumerate(lines):
        if re.match('^\s*\n$', line):
            cnt += 1
            if cnt >= 3:
                lines[i] = ''
                modified = True
        else:
            cnt = 0

    if modified:
        lines = [line for line in lines if line != '']
    
    if not outside:
        print(f'[WARN] {file} unmatched $$, no modification is done.')
    else:
        if modified:
            with open(file, 'w') as f:
                f.writelines(lines)
            print(f'[INFO] {file} fixed.')