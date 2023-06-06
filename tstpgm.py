import re
import sys

import pkg_resources


def main():


    installed_packages = [package.key for package in pkg_resources.working_set]

    print(installed_packages)

if __name__ == "__main__":
    main()