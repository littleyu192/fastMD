"""Edge-level / elementwise kernel fusions for MACE (workstream C).

* ``edge_fusions``       : ``EdgeEmbedding`` (fused edge geometry + SH + radial basis (+Agnesi)
                            + cutoff (+ZBL) (+density) with analytic Triton backward), standalone
                            fused ZBL and density-tail ops, and the unfused torch references.
* ``triton_edge_embed``  : the Triton kernels behind ``EdgeEmbedding``.
* ``triton_edge_tail``   : small Triton kernels for the MACE density tail (tanh(d^2) + scatter).

``gemm_fusions`` (if present) belongs to the cancelled GEMM workstream and is not used here.
"""
