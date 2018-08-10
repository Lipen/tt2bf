from .printers import log_error, log_warn

__all__ = ['NotBool', 'closed_range', 'b2s', 's2b', 'parse_raw_assignment_bool', 'parse_raw_assignment_int']


class NotBoolType:
    def __bool__(self):
        raise RuntimeError('this is not bool')

    def __repr__(self):
        return 'NotBool'


NotBool = NotBoolType()


def closed_range(start, stop=None, step=1):
    if stop is None:
        return range(start + 1)
    else:
        return range(start, stop + 1, step)


def b2s(data, *, s_True='1', s_False='0'):
    '''Converts 0-based bool array to string'''
    return ''.join(s_True if x else s_False for x in data)


def s2b(s, zero_based=False):
    '''Converts string to bool array'''
    ans = [{'1': True, '0': False}[c] for c in s]
    if zero_based:
        return ans
    else:
        return [NotBool] + ans


def parse_raw_assignment_int(raw_assignment, data):
    if isinstance(data[1], (list, tuple)):
        return [None] + [parse_raw_assignment_int(raw_assignment, x) for x in data[1:]]
    else:
        for i, x in enumerate(data):
            if x is not None and raw_assignment[x] > 0:
                return i
        log_warn('data[...] is unknown')


def parse_raw_assignment_bool(raw_assignment, data):
    if isinstance(data[1], (list, tuple)):
        return [None] + [parse_raw_assignment_bool(raw_assignment, x) for x in data[1:]]
    else:
        if data[0] is None:
            return [NotBool] + [raw_assignment[x] > 0 for x in data[1:]]
        else:
            return [raw_assignment[x] > 0 for x in data]
