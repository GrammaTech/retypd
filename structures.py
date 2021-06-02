'''Binary structure analysis

Using outputs from re_facts datalog, hypothesize about the structure of data in a binary.

author: Peter Aldous
'''

from collections import namedtuple
from pathlib import Path
import sys


MAY_FLOW = 'p.ap_may_flow_to.csv'
GLOBAL_ACCESS = 'p.global_accesses.csv'
PARAM_ACCESS = 'p.param_accesses.csv'


Address = namedtuple('Address', ['name', 'path'])
RawAccess = namedtuple('RawAccess', ['address', 'rw', 'size'])
Suffix = namedtuple('Suffix', ['offset', 'size'])


class AccessData:
    '''TODO'''
    def __init__(self, rw=None, size=None):
        if (rw is None) != (size is None):
            raise ValueError("Either specify an access or don't")
        self.rws = set()
        self.sizes = set()
        if rw is not None:
            self.access(rw, size)

    def access(self, rw, size):
        self.rws.add(rw)
        self.sizes.add(size)


class Structure:
    '''TODO'''
    def __init__(self, prefix, fields):
        self.address = prefix
        self.fields = fields

    def gaps(self):
        '''Find all of the gaps in the struct'''

    def overlaps(self):
        '''Find all of the fields that overlap'''


def parse_chain(chain):
    if len(chain) < 2 or chain[0] != '[' or chain[-1] != ']':
        raise ValueError(f'Bad chain: {chain}')
    return tuple(filter(lambda n: n > -1, map(int, chain[1:-1].split(', '))))


def parse_access(line):
    (name, rw, chain, size) = line.strip().split('\t')
    if rw not in {'r', 'w'}:
        raise ValueError(f'unknown read/write string: {rw}')
    access_path = parse_chain(chain)
    return RawAccess(Address(name, access_path), rw, int(size))


def get_accesses(pack_dir, binaries):
    for binary in binaries:
        global_access_path = Path(pack_dir) / binary / GLOBAL_ACCESS
        accesses = []
        try:
            with open(global_access_path, 'r') as globs:
                for glob in globs:
                    try:
                        accesses.append(parse_access(glob))
                    except ValueError:
                        print(f'Bad access record: {glob}')
                        continue
        except FileNotFoundError:
            sys.exit(f'Unable to find file {global_access_path}')
        return accesses


def print_accesses(accesses):
    accesses_by_address = {}
    for access in accesses:
        accesses_by_address.setdefault(access.address, AccessData()).access(access.rw, access.size)
    for address, data in accesses_by_address.items():
        if len(data.sizes) > 1:
            print(f'{address}: {data.rws}; {data.sizes}')


def get_structs(accesses):
    groups = {}
    for access in accesses:
        if access.address.path:
            prefix = Address(access.address.name, access.address.path[:-1])
            suffix = Suffix(access.address.path[-1], access.size)
            groups.setdefault(prefix, set()).add(suffix)
    return [Structure(prefix, sorted(suffixes)) for prefix, suffixes in groups.items()]


def usage():
    sys.exit('Usage: TODO')


def main(pack_dir, binaries):
    accesses = get_accesses(pack_dir, binaries)
    print_accesses(accesses)

    structs = get_structs(accesses)
    for struct in structs:
        if len(struct.fields) > 1:
            print(struct.address)
            for field in struct.fields:
                print(f'\t{field}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
    main(sys.argv[1], sys.argv[2:])
