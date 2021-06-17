A tool for recovering type information in binaries. It is designed to be capable of using information from re_facts, ddisasm, p-code, or other disassembly solutions. This analysis is in active development and the details of its API are expected to change.

Intended use is best demonstrated in `test/test_schema.py`. Create a `ConstraintSet` object and populate it with facts from the disassembled binary. There will eventually be a one-step solver, but the steps shown in the different tests demonstrate how to proceed.
