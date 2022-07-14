

# 0.3

- Implement `InferShapes` as in the paper to propagate all type capabilities.
- Two implementations of constraint simplification: naive path exploration and using path expressions.
- Reworked anonymous variable introduction to keep recursive type information.
- Constraint simplification is used to infer type schemes and to generate primitive constraints.
- Fixes in constraint graph generation:
    - only add recall edges and for left-hand sides of constraints and forget edges for right-hand sides
    - Add Left/Right marking to interesting nodes.

# 0.2
