import csv
import argparse

import datasets
import util

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

data = datasets.quora_dataset(args.input)

with open(args.output, 'wb') as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')

    for sample in data:
        writer.writerow([sample[3].strip('"'), sample[4].strip('"'), sample[5]])