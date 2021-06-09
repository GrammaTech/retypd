'''Binary structure analysis

Using outputs from re_facts datalog, hypothesize about the structure of data in
a binary.

author: Peter Aldous



Caveat emptor (from Drew DeHaas):

(-1, -1, -1) can store a value (or an address) for params.  but (-1, -1, -1)
must be an address for globals, and it is a fixed one (cannot be modified).

I think it is unnecessary. Your inference should always infer that (-1, -1, -1)
is a pointer for globals, which is fine (we can think of the address as being a
fixed pointer).
'''

from collections import namedtuple
from os import linesep
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
    '''A recovered data structure from the binary. Its address is an
    :class:`Address` object, which specifies its name and access path. It also
    contains a list of :class:`Suffix` objects which represent its fields.

    :param prefix: The location of the structure.
    :type prefix: class:`Address`
    :param fields: A collection of :class:`Suffix` objects, each of which
        identifies one of the structure's fields.
    :type fields: iterable
    '''
    def __init__(self, prefix, fields):
        if not isinstance(prefix, Address):
            raise ValueError(f'prefix must be an Address; got {type(prefix)}')
        self.address = prefix
        field_types = {type(f) for f in fields}
        if field_types != {Suffix}:
            raise ValueError('fields must contain only Suffix objects; '
                             f'got these types: {field_types}')
        self.fields = sorted(fields)

    def anomalies(self, identify_padding=True):
        '''Find all of the gaps in the struct as well as fields that overlap'''
        last_field = None
        overlaps = []
        reinterps = {}
        gaps = []
        padding = []
        for field in self.fields:
            if last_field:
                # upper_limit = (8 * last_field.offset) + last_field.size
                upper_limit = last_field.offset + last_field.size
                disparity = upper_limit - field.offset
                if disparity > 0:
                    overlap = (last_field, field, disparity)
                    if last_field.offset == field.offset:
                        sizes = reinterps.setdefault(field.offset, set())
                        sizes |= {last_field.size, field.size}
                    else:
                        overlaps.append(overlap)
                if disparity < 0:
                    disparity = -disparity
                    gap = (last_field, field, disparity)
                    if identify_padding and (disparity < field.size):
                        padding.append(gap)
                    else:
                        gaps.append(gap)
            last_field = field
        return gaps, reinterps, overlaps, padding

    def size(self):
        if self.fields:
            last_field = self.fields[-1]
            return last_field.offset + last_field.size
        return 0

    def __str__(self):

        def stringify(label, empty_message, values, formatter=str):
            if values:
                value_strings = map(lambda v: f'\t\t{formatter(v)}', values)
                all_values = linesep.join(value_strings)
                return f'{linesep}\t{label}:' + linesep + all_values
            else:
                if empty_message:
                    return f'{linesep}\t{empty_message}'
                else:
                    return ''

        def bytes_label(bytes):
            label = f'{bytes} bytes'
            if bytes == 1:
                return label[:-1]
            return label

        def span_formatter(span):
            before, after, size = span
            return (f'{size} bytes between fields at offset {before.offset} '
                    f'({bytes_label(before.size)}) and offset {after.offset} '
                    f'({bytes_label(after.size)})')

        def field_formatter(field):
            return (f'field at offset {field.offset} '
                    f'({bytes_label(field.size)})')

        def reinterpretation_formatter(reinterpretation):
            offset, sizes = reinterpretation
            sizes = sorted(sizes)
            if len(sizes) == 2:
                sizes_str = f'{sizes[0]} and {sizes[1]}'
            else:
                others = sizes[:-1]
                last = sizes[-1]
                sizes_str = ', '.join(map(str, others)) + f', and {last}'
            return f'field at offset {offset} is read for {sizes_str} bytes'

        def address_formatter(address):
            path_str = ''.join(map(lambda offset: f'[{offset}]', address.path))
            return address.name + path_str
        result = (f'Struct at {address_formatter(self.address)} '
                  f'({bytes_label(self.size())}):')
        gaps, reinterps, overlaps, padding = self.anomalies()
        result += stringify('fields', None, self.fields, field_formatter)
        result += stringify('gaps', None, gaps, span_formatter)
        result += stringify('padding', None, padding, span_formatter)
        result += stringify('overlaps', None, overlaps, span_formatter)
        result += stringify('reinterpretations',
                            None,
                            reinterps.items(),
                            reinterpretation_formatter)
        return result

    def __repr__(self):
        return str(self)


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


def print_accesses(accesses, access_filter=None):
    accesses_by_address = {}
    for access in accesses:
        accesses = accesses_by_address.setdefault(access.address, AccessData())
        accesses.access(access.rw, access.size)
    for address, data in accesses_by_address.items():
        if access_filter is None or access_filter(data):
            print(f'{address}: {data.rws}; {data.sizes}')


def get_structs(accesses, include_reinterps=False):
    groups = {}
    for access in accesses:
        if access.address.path:
            prefix = Address(access.address.name, access.address.path[:-1])
            suffix = Suffix(access.address.path[-1], access.size)
            groups.setdefault(prefix, set()).add(suffix)
    return [Structure(prefix, suffixes) for prefix, suffixes in groups.items()
            if include_reinterps or
               (len({suffix.offset for suffix in suffixes}) > 1)]


def usage():
    sys.exit('Usage: TODO')


def main(pack_dir, binaries):
    accesses = get_accesses(pack_dir, binaries)
    print_accesses(accesses, lambda data: len(data.sizes) > 1)
    print()

    structs = get_structs(accesses)
    for struct in structs:
        if len(struct.fields) > 1:
            print(struct)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
    main(sys.argv[1], sys.argv[2:])
