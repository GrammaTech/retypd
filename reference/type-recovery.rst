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
discussions with a collaborator on the project suggest that "structural subtyping" may to refer to
`depth subtyping <https://en.wikipedia.org/wiki/Subtyping#Width_and_depth_subtyping>`_ and
"non-structural subtyping" to width subtyping (same link as depth). We're not certain what "physical
subtyping" means.

------------------------
Non-structural subtyping
------------------------

Throughout the paper, the subset relation (⊑) is used simply to mean that one datum's type is a
subtype of another. Most frequently, it is used to model depth (structural) subtyping. However, at
least one occurrence (p.7, α ⊑ F.in_stack0) indicates width (non-structural) subtyping. The paper
states on multiple occasions that instantiation, such as on call site boundaries, allows this
non-structural subtyping to occur.

The key is not function calls; nothing in the analysis makes function calls different from other
components of a program. However, using functions as "interesting" variables does create boundaries.
For example, a path in the graph that reaches some function F via its in_0 capability will generate
a constraint whose right-hand side begins with "F.in_0", indicating something about the type of a
value passed to F. A path from F to something else via its in_0 capability will generate a
constraint whose left-hand side begins with "F.in_0". Because these type constraints belong to
different sketches, any free type variables are implicitly instantiated at this boundary.

This non-structural subtyping can skip these boundaries, but only in well-defined ways. For example,
if F accepts a struct with two members and G passes a compatible struct that adds a third member, it
is possible for G's sketch to include information from path downstream from the F.in_0 vertex it
encounters. However, it can only infer things about the first two members from the uses in F; F
cannot infer anything about the third member because it would require going back up the graph.
(IS THIS TRUE? CAN WE PROVE IT FOR ALL CASES?)

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

Once the initial graph is created, edges corresponding to S-Field⊕ and S-Field⊖ are added as
shortcuts between existing nodes. The lazy instantiation of S-Pointer is presented in Algorithm D.2,
lines 20-27. It is also shown in Figure 14 (in one of two possible forms). The form shown uses an
instantiation of S-Field⊖ for a store, then an instantiation of S-Pointer, and finally an
instantiation of S-Field⊕ for a store. The rightmost edge in Figure 14 shows the contravariant edge
generated (lazily) by S-Pointer that would be used if the load and store were swapped. In either
case, the first instantiation requires three or more edges and the last of these edges must be a pop
(or recall) edge. The head of the pop edge always comes from a node with a contravariant elided
prefix (in the figure, p⊖). The target of the first edge required by the last instantiation is
always a node with the same derived type variable but with inverted variance (in the figure, p⊕).
N.B. this triple instantiation of rules does not create any new nodes.

As a result, saturation adds edges for S-Field⊕ and S-Field⊖ between nodes that already exist. It
also adds edges for S-Pointer (combined with the other two rules). This limits these rules'
instantiations so that they never create additional nodes in the graph. As a result, saturation
converges. I have not yet proven that this guarantees that all useful instantiations of these rules
occur in this limited context, but I think that the proof in Appendix B proves this property.

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
   create an edge from p to r without a label. Repeat to a fixed point. Additionally, create
   shortcut edges as shown in Figure 14 for S-Field/S-Pointer/S-Field instantiations.
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
#. If desired, generate sketches from the type constraints.
