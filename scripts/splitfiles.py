"""
splits files in 'source' directory by lines into 'inputs' directory
"""


import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    source_path = os.path.join('data', args.dataset, 'source')
    inputs_path = os.path.join('data', args.dataset, 'inputs')
    if not os.path.isdir(inputs_path):
        os.mkdir(inputs_path)

    all_lines = set()
    for name in os.listdir(source_path):
        source_file_path = os.path.join(source_path, name)
        lines = []
        with open(source_file_path) as f:
            data = f.read()
        for line in data.split('\n'):
            if line not in all_lines:
                lines.append(line)
        print(f'writing {len(lines)} lines from {name}')
        for i in range(len(lines)):
            with open(os.path.join(inputs_path, f'{name}-{i}.txt'), 'w') as f:
                f.write(lines[i])
            print(f'\r({"%" * int(100.0 * i / len(lines))}{" " * int(100 - 100.0 * i / len(lines))})', end='')
        print()


if __name__ == '__main__':
    main()
