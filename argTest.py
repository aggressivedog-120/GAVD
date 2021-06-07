import argparse

parser = argparse.ArgumentParser()
parser.description = "arguments for stable learning"
parser.add_argument("--n", help="number of all generated samples", type=int)
parser.add_argument("--p", help="number of features", type=int)
parser.add_argument("--r", help="bias rate of bias selection", type=int)
args = parser.parse_args()

if args.n:
    print('n={}'.format(args.n))
if args.p:
    print('p={}'.format(args.p))
print("times: {}".format(args.n * args.p))





