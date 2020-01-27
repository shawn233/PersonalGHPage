from __future__ import print_function
import os
import sys

PWD = os.getcwd()
target_filename = "2020-01-24-CS:GO---Quick-Configuration.md.hidden"
target_file = os.path.join(PWD, "..", target_filename)

def main():
    included_bash_cnt = {2, 3}

    bash_cnt = 0
    in_bash = False
    with open(target_file) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0:2] == "//":
                continue
            elif line[0:3] == "```":
                if "bash" in line:
                    bash_cnt += 1
                    in_bash = True
                else:
                    in_bash = False
            elif in_bash and bash_cnt in included_bash_cnt:
                ind = line.find("//")
                if ind == -1:
                    print(line, end=";")
                else:
                    print(line[:ind].strip(), end=";")
    print("\n\nDone!")

if __name__ == "__main__":
    main()
