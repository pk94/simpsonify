import csv
import argparse
from pathlib import Path


def generate_metafile(path):
    classes = []
    with open('metafile.csv', mode='w', newline='') as metafile:
        metafile_writer = csv.writer(metafile, delimiter=',')
        metafile_writer.writerow(['label', 'path'])
        for subpath in path.iterdir():
            classes.append(subpath.name)
            for file in subpath.rglob("*"):
                if file.is_file():
                    metafile_writer.writerow([subpath.name, file])


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data_path", type=Path, default=f'C:\\Pawel\\Datasets\\human2simpson')
args = parser.parse_args()
generate_metafile(args.data_path)
