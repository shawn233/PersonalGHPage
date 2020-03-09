#encoding: utf-8

import argparse

def main():
    parser = argparse.ArgumentParser(description="a simple scipt to repeat words")
    parser.add_argument("word", help="the word to repeat", type=str)
    parser.add_argument("-n", "--number", help="repeat the word for <number> times", dest="number", type=int, default=5)
    parser.add_argument("-s", "--sep", help="seperators inserted between every word pairs", dest="seperator", type=str, default=" ")
    args = parser.parse_args()

    assert(len(args.word) > 0)
    assert(args.number >= 0)

    if (args.number == 0):
        return

    for _ in range(1, args.number):
        print(args.word, end=args.seperator)
    print(args.word)

if __name__ == "__main__":
    main()
