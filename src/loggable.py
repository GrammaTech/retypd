from enum import Enum


class LogLevel(int, Enum):
    QUIET = 0
    INFO = 1
    DEBUG = 2

# Unfortunable, the python logging class is a bit flawed and overly complex for what we need
# When you use info/debug you can use %s/%d/etc formatting ala logging to lazy evaluate
class Loggable:
    def __init__(self, verbose: LogLevel = LogLevel.QUIET):
        self.verbose = verbose

    def info(self, *args):
        if self.verbose >= LogLevel.INFO:
            print(str(args[0]) % tuple(args[1:]))

    def debug(self, *args):
        if self.verbose >= LogLevel.DEBUG:
            print(str(args[0]) % tuple(args[1:]))
