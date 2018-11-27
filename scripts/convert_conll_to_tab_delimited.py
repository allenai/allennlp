import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

import argparse

def main(inp_fn: str,
         out_fn: str) -> None:
    """
    inp_fn: str, required.
       Path to file from which to read CoNLL with space delimited fields.
    out_fn: str, required.
       Path to file to which to write the CoNLL format Open IE extractions.
    """
    with open(out_fn, 'w') as fout:
        for line in open(inp_fn):
            if line.startswith("#") or not(line.strip()):
                fout.write(line)
            else:
                fout.write("\t".join(line.strip().split()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Open IE4 extractions to CoNLL (ontonotes) format.")
    parser.add_argument("--inp", type=str, help="input file from which to read CoNLL with space delimited fields.", required = True)
    parser.add_argument("--out", type=str, help="path to the output file, where CoNLL format should be written.", required = True)
    args = parser.parse_args()
    main(args.inp, args.out)

