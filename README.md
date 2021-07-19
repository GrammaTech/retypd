A tool for recovering type information in binaries. It is designed with a simple front end schema so that it can be used with any disassembler. **This analysis is in active development and the details of its API are expected to change.**

Intended use is best demonstrated in `test/test_schema.py`'s `test_end_to_end` method: Create a `ConstraintSet` object and populate it with facts from the disassembled binary. Instantiate a Solver object with the populated ConstraintSet and a collection of interesting variables (as strings or DerivedTypeVariable objects). Call the Solver object (it needs no arguments after it is instantiated). Inferred constraints are stored in an attribute called constraints.

Several additional details are included in `reference/type-recovery.rst`, including explanations of some of the more complex concepts from the paper and an outline of the type recovery algorithm.

## Copyright and Acknowledgments

Copyright (C) 2021 GrammaTech, Inc.

This code is licensed under the GPLv3 license. See the LICENSE file in the project root for license terms.

This project is sponsored by the Office of Naval Research, One Liberty Center, 875 N. Randolph Street, Arlington, VA 22203 under contract #N68335-17-C-0700. The content of the information does not necessarily reflect the position or policy of the Government and no official endorsement should be inferred.
