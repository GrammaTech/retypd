# Retypd - machine code type inference
# Copyright (C) 2021 GrammaTech, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This project is sponsored by the Office of Naval Research, One Liberty
# Center, 875 N. Randolph Street, Arlington, VA 22203 under contract #
# N68335-17-C-0700.  The content of the information does not necessarily
# reflect the position or policy of the Government and no official
# endorsement should be inferred.

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
