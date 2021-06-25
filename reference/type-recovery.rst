***************************************
Type inference (in the style of retypd)
***************************************

Several people have implemented or attempted to implement retypd. It is a powerful system but
difficult to understand. My goal in this document is to explain what I have learned as I have
studied the paper and implemented the analysis.

References to the paper are to the extended form which is included in this directory and available
on `ArXiv <https://arxiv.org/pdf/1603.05495.pdf>`_. References to the slides, which are also
included in this directory, are from `Matt's PLDI talk
<https://raw.githubusercontent.com/emeryberger/PLDI-2016/master/presentations/pldi16-presentation241.pdf>`_.
The slides, as a PDF, lack animations and I will reference them by "physical" number, not by the
numbers printed in the slides themselves.

###############
What is retypd?
###############

Retypd is a polymorphic type inference scheme, suitable for type recovery from binaries. Its primary
contribution is a subtlety in how it treats pointer types; some uses of pointers are covariant and
some uses are contravariant. By managing this variance effectively, it is able to reason much more
precisely than unmodified unification-based type inference algorithms.

Retypd was originally published and presented at PLDI 2016 by Matthew Noonan, who was a GrammaTech
employee at the time. His presentation is very dense and subtle (but quite rigorous). This document
hopes to distill some of the details learned by studying the paper, the presentation, and by
creating an implementation.

#############################
Subtleties and misconceptions
#############################

-----------
Scaffolding
-----------

Many of the mathematical constructs in this paper exist for scaffolding; they develop the reader's
intuition about how the operations work. However, they are not actually used in the algorithm. The
steps outlined in the final section summarize the portions of the paper and slides that actually
should be implemented.

----------------
Subtyping styles
----------------

The paper asserts things about several subtyping modalities. While I cannot speak for the author,
Colin Unger (who implemented much of retypd in Datalog previously) believes that the paper uses
"structural subtyping" to refer to `depth subtyping
<https://en.wikipedia.org/wiki/Subtyping#Width_and_depth_subtyping>`_, "non-structural subtyping" to
refer to width subtyping (same link as depth), and we're not certain what "physical subtyping"
means.

-----------------
Equivocation on ⊑
-----------------

The use of the subset relation (⊑) is usually part of type constraints (depth subtyping, per my
conversations with Colin). It is also used in one location to refer to width subtyping (p.7, α ⊑
F.in_stack0). The distinction the authors draw is that one style of subtyping occurs in the type
constraints and the other occurs in specialization at function call boundaries (an arbitrary but
pragmatic design decision).

What wasn't immediately clear to me is that the type inference rules in Figure 3 are applied to a
fixed point before the "real" inference begins (p. 21 - "suppose we have a *fixed* set of
constraints C over V", emphasis mine).

In many type inference systems, the inference is a fixed point over a set of constraints from a set
of input facts. Retypd does the same, but this fixed point is a preliminary step. Its inference
rules like T-InheritR (which, together with T-InheritL, does depth subtyping) have reached a fixed
point before any of the width subtyping rules are applied at a later phase.

-----------------
Variance notation
-----------------

The notation that writes variance as if it were a capability (e.g., F.⊕) indicates not the variance
of the variable that precedes it rather summarizes the variance of everything on the stack *after*
F. In other words, it is the variance of the elided portions of the stack.

For the formation of the graph (shown in slides 68-78), it is especially important to remember that
the variance of the empty string (⟨ε⟩) is ⊕. Since type constraints from C elide no part of their
stack, the elided portion is ε. In other words, all of the symbols copied directly from C into nodes
in the graph have variance ⊕.

-------------------
Special graph nodes
-------------------

Start# and End# and the L and R subscripts are conveniences for ensuring that we only examine paths
from interesting variables to interesting variables whose internal nodes are exclusively
*uninteresting* variables. For many implementations, there is no need to encode these things
directly.

------------------------------
Finding the graph in the paper
------------------------------

The slides use a different notation than does the paper. The graph construction summarized in slides
68-78 corresponds to Appendices C and D (pp. 20-24).

----------------
The type lattice
----------------

The lattice Λ of types is somewhat arbitrary, but C types are too general to be useful. Instead, the
TSL retypd implementation uses types as users think about them. A file descriptor, for example, is
implemented with an int but is conceptually different. It ought to be possible to recover the basic
types from the TSL code.

The two JSON files in this directory include the schemas from the TSL implementation. The presence
of any functions included in these schemas will yield types with semantic meaning.

------------------------
S-Pointer and S-Field⊕/⊖
------------------------

Most of the inference rules in Figure 3 (p. 5) become redundant because of the structure of the
graph:
* T-Left and T-Right are implicit because edges cannot exist without vertices.
* T-Prefix is expressed by the existence of forget edges.
* As discussed in the paper, T-InheritL and T-InheritR are redundant, as a combination of
  T-Left/T-Right and S-Field⊕/S-Field⊖ can produce the same facts.
* S-Refl and S-Trans are implicit.

S-Pointer, S-Field⊕, and S-Field⊖ are not expressed in the graph. They are also problematic because
they can create divergence (this is true of all three inference rules, not just S-Pointer). The
paper explicitly explains that S-Pointer rules are instantiated lazily. However, it does not explain
that the instantiations in Algorithm D.2 (p. 25) are actually instantiations of all three of these
rules combined into one. The algorithm as written is safe; it is not yet perfectly clear to me
whether there are useful things that might be inferred from other instantiations of the S-Field
rules.

#####################
Type recovery outline
#####################

The following steps, in order, implement retypd. The steps after saturation reflect recent updates
to retypd and not the original paper.

#. Generate base constraints (slides 18-27 or Appendix A). Call this set of constraints C.
#. Do **not** fix the set of constraints over the inference rules from Figure 3 (see also slide 28);
   this diverges in the presence of recursive types. The remainder of the algorithm accomplishes the
   same thing as the fixed point but without diverging.
#. Build a graph Δ from C; a ⊑ b becomes a.⊕ → b.⊕ *and* b.⊖ → a.⊖ (Δ_c on p. 21). Each of these
   edges is unlabeled.
#. For every node with capabilities (e.g., a.c.⊕), create "forget" and "recall" edges. For our
   example node, let us assume that c is contravariant (i.e., ⟨c⟩ = ⊖). Produce an edge with the
   label "forget c" from a.c.⊕ → a.⊖ and an edge with the label "recall c" in the opposite
   direction. This may or may not create additional nodes. Forget and recall edges are used in the
   slides and, respectively, are called push and pop edges in the paper (see step 2 of D.2 on page
   22). **N.B. forgetting is equated with pushing because the elided capability is pushed onto the
   stack.**
#. Saturate by finding *sequences* of edges that are all unlabeled except for a single forget edge
   (say, "forget *l*") that reach nodes with outgoing edges with a corresponding recall edge
   ("recall *l*"). If the sequence begins and reaches q and if the recall edge is from q to r,
   create an edge from p to r without a label. Repeat to a fixed point. During this phase, consider
   every pair of nodes of the form (v.store.⊕, v.load.⊕) or (v.load.⊖, v.store.⊖) to be connected by
   an implicit edge without a label.
#. Remove self loops; the graph represents a reflexive relation, so edges from a vertex to itself
   are not informative.
#. Identify cycles (strongly connected components) in the graph that do not include both forget and
   recall edges. Identify nodes in these cycles that have predecessors outside of the SCC. Eliminate
   duplicates (there is no need to include A.load if A is already in the set). Create a new type
   variable for each remaining node and add each of these nodes to the set of interesting variables.
#. Split the graph into two subgraphs, copying recall and unlabeled edges but not forget edges to
   the new subgraph. Change the tails of existing recall edges to the nodes in the new subgraph.
   This ensures that paths can never include forget edges after recall edges.
#. Starting at each node associated with an interesting variable, find paths to other interesting
   variables. Record the edge labels. For each path found, generate constraints: append the forget
   labels to the interesting variable at the beginning of the path and the recall labels to the
   interesting variable at the end of the path. If both of the resulting derived type variables have
   a covariant suffix and if they are not equal to each other, emit a constraint.
