# encoding: utf-8
# file: foo.py

from __future__ import print_function
import sys

def main():
    for i in range(len(sys.argv)):
        print("argv[{}]: {}".format(i, sys.argv[i]))

main()
