import csv
from pathlib import Path

def generate_metafile(path):
    path = Path(path)
    classes = []
    with open('metafile.csv', mode='w', newline='') as metafile:
        metafile_writer = csv.writer(metafile, delimiter=',')
        metafile_writer.writerow(['label', 'path'])
        for subpath in path.iterdir():
            classes.append(subpath.name)
            for file in subpath.rglob("*"):
                if file.is_file():
                    metafile_writer.writerow([subpath.name, file])

generate_metafile(f'C:\Pawel\Datasets\human2simpson')