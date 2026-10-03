# Related work

The failure mode this repository studies is established in the cheminformatics
literature. Nothing here is presented as a novel finding about optimizer
failure, and H1, H2 and H4 are replications.

## Prior work

**Renz, Rak, Hochreiter & Klambauer (2019), "On failure modes in molecule
generation and optimization"**, *Drug Discovery Today: Technologies* 32-33,
55-63. Optimization scores climb while control scores, computed with models
trained on held-out data, stagnate at far lower values. Their experimental
design -- an optimization model, a *model-control* differing in architecture,
and a *data-control* trained on a disjoint split -- is the methodological
ancestor of this project's H3 and is adopted directly in `docs/BRIEFB-V2.md`
SS4.2. See `docs/LIMITATIONS.md` for the respect in which this repository's
`data_control_half` falls short of their design, and why.

**Langevin, Bianciotto, Vuilleumier, Minoux & Schwartz (2022), "Explaining and
avoiding failure modes in goal-directed generation of small molecules"**,
*Journal of Cheminformatics* 13:83, doi:10.1186/s13321-022-00601-y. Reproduces
Renz across Graph GA, SMILES-LSTM and MSO, and analyses causes. Three optimizer
families are therefore already covered in the literature, which is why a second
optimizer is deprioritized here.

**Gao & Coley (2020), "The Synthesizability of Molecules Proposed by Generative
Models"**, *Journal of Chemical Information and Modeling* 60(12), 5714-5723,
arXiv:2002.07007. Quantifies how often generated molecules cannot be readily
synthesized, and shows that synthetic-complexity heuristics bias generation
toward tractable space but necessarily detract from the primary objective.
**H4 is a replication of this result on a different objective.**

## What is new here

Only the following may be framed as contributions of this repository:

1. A per-constraint ablation on a *shared* composite objective, yielding a
   quantified cost curve (objective score retained vs. implausibility removed)
   per constraint, rather than a single synthesizability result.
2. Pre-registered implausibility thresholds, calibrated against approved drugs,
   fixed before the first full run. See `docs/PREREGISTRATION.md`.
3. A multi-objective ADMET triage objective, rather than the single-target
   activity QSAR attacked in the prior work.
4. Per-generation divergence tracking, which prior work reports only as endpoint
   trajectories of the control scores, not as an operational monitoring signal.

## Framing

This project examines naive composite objective design. ADMET-AI and TDC are
well-built resources being deliberately used outside their intended operating
regime; nothing here is a criticism of those tools or their authors.

Molecules produced by the optimizer are artifacts of an adversarial search
procedure. They are not drug candidates, hits or leads, and no claim is made
here about clinical, toxicological or regulatory implications.
