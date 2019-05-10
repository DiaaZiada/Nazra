import argparse

from Agent import *


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="Nazra Configuration")
    arg_parser.add_argument('-c',
                            '--config',
                            metavar='config_json_file',
                            default='None',
                            help='The Configuration file in json format')
    args = arg_parser.parse_args()


if __name__ == '__main__':
    main()