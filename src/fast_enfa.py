from pyformlang.finite_automaton import (
    EpsilonNFA,
    State,
    DeterministicFiniteAutomaton,
)
from typing import Iterable


def to_single_state(l_states: Iterable[State]) -> State:
    """
    Merge a list of states
    """
    values = [str(state.value) if state else "TRASH" for state in l_states]
    return State(";".join(values))


class FastENFA(EpsilonNFA):
    def _to_deterministic_internal(
        self, eclose: bool
    ) -> DeterministicFiniteAutomaton:
        """
        Transforms the epsilon-nfa into a dfa

        NOTE: This a a modified version of the original EpsilonNFA._to_deterministic_internal
        This has a few small changes:
        - Refactored the call to `add_final_state` to be less redundant
        - Added some additional checks to `all_trans` computation to filter out invalid items
        """
        dfa = DeterministicFiniteAutomaton()
        # Add Eclose
        if eclose:
            start_eclose = self.eclose_iterable(self._start_state)
        else:
            start_eclose = self._start_state

        start_state = to_single_state(start_eclose)

        dfa.add_start_state(start_state)
        to_process = [start_eclose]
        processed = {start_state}

        while to_process:
            current = to_process.pop()
            s_from = to_single_state(current)

            if any(state in self._final_states for state in current):
                dfa.add_final_state(s_from)

            for symb in self._input_symbols:
                all_trans = [
                    self._transition_function._transitions[x][symb]
                    for x in current
                    if (
                        x in self._transition_function._transitions
                        and symb in self._transition_function._transitions[x]
                    )
                ]
                states = set()
                for trans in all_trans:
                    states = states.union(trans)
                if not states:
                    continue
                # Eclose added
                if eclose:
                    states = self.eclose_iterable(states)
                state_merged = to_single_state(states)
                dfa.add_transition(s_from, symb, state_merged)
                if state_merged not in processed:
                    processed.add(state_merged)
                    to_process.append(states)

        return dfa
