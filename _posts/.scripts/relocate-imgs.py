#encoding: utf-8
import re
import os


ONLINE_PREFIX = r"https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/"
LOCAL_IMG = r"!\[.*\]\(\.assets/[\w_.%-]+\.(?:png|jpg|jpeg)\)"


def main():
    '''detect all local image paths in texts, and replace them with online paths'''
    pat = re.compile(LOCAL_IMG)

    target_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..")
    print(f"Article directory: {target_dir}")

    files = os.listdir() # article directory
    for filename in files:
        if not filename.endswith(".md"):
            continue

        print(f"\nProcessing file: {filename} ...")
        file_data = ""
        changed = False
        with open(os.path.join(target_dir, filename), "r") as f:
            for idx, line in enumerate(f):
                mat = pat.search(line)
                if mat is None:
                    file_data += line
                    continue
                
                changed = True
                new_line = line.replace(".assets/", ONLINE_PREFIX)
                file_data += new_line

                print(f"Line {idx}, changing\n\t {line}\t->\n\t {new_line}")
        
        if changed:
            print(f"File {filename} is changed, original file saved at {filename+'.bak'}")
            with open(os.path.join(target_dir, filename+".online"), "w") as f:
                f.write(file_data)
            
            os.rename(
                os.path.join(target_dir, filename),
                os.path.join(target_dir, filename+".bak"))
            os.rename(
                os.path.join(target_dir, filename+".online"),
                os.path.join(target_dir, filename))


if __name__ == "__main__":
    main()
