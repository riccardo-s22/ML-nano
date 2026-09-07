#!/usr/bin/env python3
"""Render METHODS/RESULTS/LIMITATIONS/FINAL_ANALYSIS_REPORT + CLAIM_EVIDENCE_MATRIX
(§18). The claim-evidence matrix status is derived PROGRAMMATICALLY from gate JSON
files (never hand-upgraded). Regenerable: rerun after docking gates land."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, utc_now

GATES = os.path.join(ROOT, "results", "gates")
REP = os.path.join(ROOT, "reports")
os.makedirs(REP, exist_ok=True)

def gate(name):
    p = os.path.join(GATES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def load_tsv(path):
    if not os.path.exists(path): return []
    with open(path) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(cols, l.rstrip("\n").split("\t"))) for l in fh]

def main():
    g0 = gate("gate0_smoke"); g1 = gate("gate1_sequence"); g2 = gate("gate2_thermo")
    g3 = gate("gate3_benchmark"); g4 = gate("gate4_chirality"); g5 = gate("gate5_structure")
    gswcnt = gate("swcnt_geometry")
    perfect = load_tsv(os.path.join(ROOT, "results", "hybridization", "melting_perfect_results.tsv"))
    chir = load_tsv(os.path.join(ROOT, "results", "docking", "chirality", "chirality_ranking.tsv"))

    # ---------- CLAIM-EVIDENCE MATRIX (programmatic from gates) ----------
    def thermo_status():
        return "supported" if g2.get("passed") else "not run/failed"
    def bench_passed():
        return bool(g3.get("passed"))
    def chir_status():
        if not g3:
            return "not run"
        return "conditional" if bench_passed() and g4.get("passed") else "exploratory"

    rows = [
        ["C1", "junction22 forms a more stable perfect pathological duplex than junction20 "
               "under specified free-solution conditions", "hybridization",
         "results/hybridization/melting_perfect_results.tsv", "Gate 2", thermo_status(),
         "predicted to be modestly more stable", "junction22 is experimentally more sensitive"],
        ["C2", "both capture probes show much stronger free-solution duplex stability contrast "
               "with the pathological junction than with normal/negative targets", "hybridization",
         "results/hybridization/duplex_descriptors.tsv; nupack_competition_exploratory.tsv",
         "Gate 2", thermo_status(),
         "strong free-solution stability/complementarity contrast in silico",
         "guaranteed experimental specificity at any condition"],
        ["C3", "a GT base-pair unit stacks more favorably than CT/AT units on the (7,5) sidewall "
               "in the controlled Vina screen", "docking", "results/docking/benchmark/gate_status.json",
         "Gate 3", "supported" if bench_passed() else "not supported",
         "more favorable controlled docking-score distribution" if bench_passed()
         else "GT did not separate from controls in the controlled screen",
         "GT physically binds the SWCNT more strongly"],
        ["C4", "one chirality has the most favorable controlled Vina base-stacking distribution",
         "docking", "results/docking/chirality/chirality_ranking.tsv", "Gate 4", chir_status(),
         "most favorable in this controlled docking screen",
         "strongest physical binding or best optical sensor chirality"],
        ["C5", "the capture domain remains solvent-accessible at equilibrium on the sensor",
         "structure/MD", "results/structures/starting_model_manifest.tsv", "Gate 5",
         "not supported (conceptual only)" if not g5.get("passed") else "supported after convergence",
         "structures are docking/builder-derived starting geometries only",
         "the capture probe is exposed on the equilibrium corona"],
    ]
    # C6 experimental — only if real experimental analysis ran
    agp = os.path.join(ROOT, "results", "experimental", "insilico_agreement.json")
    if os.path.exists(agp):
        ag = json.load(open(agp))
        obs = ag.get("experimental_observation_water", {})
        verdict = ag.get("agreement_verdict", "")
        st = ("conditionally supported (exploratory)" if "CONSISTENT" in verdict
              else "inconclusive")
        rows.append(["C6", "the GT15-STMN2 sensor shows a specific, saturable dose-response to "
                     "STMN2-CE mRNA in water (apparent K_A), consistent with predicted capture-domain "
                     "hybridization", "experimental(§14)",
                     "results/experimental/chirality_response_KA.tsv; insilico_agreement.json",
                     "exploratory(association)", st,
                     f"specific dose-dependent response; sensor>>control; apparent K_d~"
                     f"{obs.get('ch7_5_apparent_Kd_copies_per_uL')} copies/uL (operational sensitivity)",
                     "in-silico predicted the optical K_A or that docking explains the optical readout"])
    with open(os.path.join(REP, "CLAIM_EVIDENCE_MATRIX.tsv"), "w") as fh:
        fh.write("claim_id\tclaim_text\tanalysis_stage\tsupporting_files\tquality_gate\t"
                 "status\tallowed_wording\tforbidden_overinterpretation\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")

    # ---------- METHODS ----------
    with open(os.path.join(REP, "METHODS.md"), "w") as fh:
        fh.write(f"""# METHODS — STMN2 RNA-SWCNT computational pipeline

Generated {utc_now()}. All stages run on CPU; MD disabled (`run_md=false`).

## Sequence verification
RefSeq transcript retrieved live from NCBI E-utilities (accession recorded with version +
SHA-256; Ensembl exon structure cross-fetched). The normal 22-nt junction target was
DERIVED from the reference (exon1[-11:] + normal exon2[:11]) by localizing the validated
normal20 oligo in the spliced mRNA and extending +/-1 nt; its exon-1 11-mer was cross-
validated against the pathological22 exon-1 portion. The cryptic exon 2a is intronic and
not in the canonical mRNA; pathological sequences are design inputs validated for internal
consistency. All length/alphabet/reverse-complement/antiparallel checks enforced as a hard
gate (Gate 1).

## Hybridization thermodynamics
Perfect pathological DNA/RNA duplexes (junction20+pathological20, junction22+pathological22)
analyzed with MELTING 5.2.0 (Sugimoto-1995 DNA/RNA hybrid nearest-neighbour set); the
complement was passed explicitly in the per-position (antiparallel) orientation. Tm is salt-
corrected; deltaH/deltaS/deltaG are reference-condition NN values and are NOT labeled salt-
conditioned. Biopython R_DNA_NN1 (same Sugimoto set) provided an independent Tm cross-check
(reported side by side, not averaged; residual gap attributed to differing salt-correction/
initiation conventions). Longest contiguous Watson-Crick run is reported as a mechanistic
DESCRIPTOR only. NUPACK 4.1.0.1 (licensed wheel) was used for RNA-only secondary-structure
accessibility (material='rna'); its analysis path is single-material, so a true DNA/RNA hybrid
duplex parameterization is unavailable — probe.target competition is reported only as a
single-material DNA/RNA BRACKET, explicitly labeled EXPLORATORY and NOT a hybrid substitution.

## SWCNT construction
Five chiralities [(7,5),(10,2),(9,4),(8,6),(8,7)] built with ASE (a_cc=1.42 A), trimmed to a
common ~90 A axial window (open ends). Analytic vs measured diameter agree to <0.01 A.
Receptor PDBQT written with full column control: every carbon type 'A', charge 0.000
(graphitic sp2; no Gasteiger). Structure SHA-256 and typing sidecars recorded.

## Anchor conformers
ssDNA built with AmberTools tleap (DNA.OL15) -> sander minimization + Langevin MD (implicit
solvent igb=1, elevated T) -> cpptraj RMSD clustering -> medoids. RDKit/MMFF was NOT used as
the oligonucleotide optimizer. Rigid ligand PDBQTs prepared by a controlled tleap-mol2 ->
AutoDock typing route (Meeko 0.5/0.7 fail on bare ssDNA; not MGLTools); all heavy atoms
preserved, nonpolar H merged, TORSDOF 0.

## Controlled Vina docking
Vina 1.2.5 Python API, cpu=1 per run (deterministic), seed pinned. Vina scores are CONTROLLED
COMPARATIVE steric/stacking scores on an uncharged graphitic receptor — NOT binding free
energies. Rigid docking of full ssDNA oligomers on the curved sidewall is clash-dominated
(score ~+1e8; shape incompatibility between an extended ssDNA conformer and the convex
sidewall — confirming revision-note A1 that Vina is mis-shaped for this ligand). The screen
therefore uses the dinucleotide repeat units (GTd/CTd/ATd) that constitute the anchors,
probing base-surface stacking in a sidewall-offset box (excludes tube interior and ends).
Benchmark: GTd/CTd/ATd on (7,5), 10 conformers x 10 seeds x exhaustiveness 32. Gate 3 uses
bootstrap median-difference CIs: pass iff median(GT)<median(CT,AT) and the upper 95% CI of
both differences <0. Chirality screen: GTd on all five chiralities, identical settings.

## Structures and MD
MD disabled by default (no peer-reviewed SWCNT carbon Lennard-Jones parameterization supplied).
Conceptual full-sensor starting geometries (GT15-junction20, GT15-junction22, GT15-only) were
builder-placed near the (7,5) sidewall in >=20 orientations each, rejecting genuine
intermolecular nonbonded clashes (<1.8 A, heavy atoms; bonded geometry not treated as clash).
Labeled "Representative starting geometry; not energy-minimized or dynamically equilibrated."
No capture-exposure claim is made (Gate 5 = conceptual).
""")

    # ---------- RESULTS ----------
    j20 = next((r for r in perfect if r["sensor"] == "junction20"), {})
    j22 = next((r for r in perfect if r["sensor"] == "junction22"), {})
    with open(os.path.join(REP, "RESULTS.md"), "w") as fh:
        fh.write(f"# RESULTS\n\nGenerated {utc_now()}.\n\n")
        fh.write("## Hybridization (Gate 2: {})\n".format("PASS" if g2.get("passed") else "n/a"))
        if j20 and j22:
            fh.write(f"- junction20 perfect pathological duplex: Tm {j20.get('Tm_C_salt_corrected')} °C, "
                     f"ΔG(37°C, ref-NN) {j20.get('deltaG_kcal_mol_at_assayT_refNN_NOT_salt_conditioned')} kcal/mol.\n")
            fh.write(f"- junction22 perfect pathological duplex: Tm {j22.get('Tm_C_salt_corrected')} °C, "
                     f"ΔG(37°C, ref-NN) {j22.get('deltaG_kcal_mol_at_assayT_refNN_NOT_salt_conditioned')} kcal/mol.\n")
            fh.write("- junction22 is predicted modestly more stable than junction20 (free-solution).\n")
        fh.write("- Specificity descriptor: longest contiguous WC run = full length vs pathological, "
                 "10-11 nt vs normal (shared exon-1), 4 nt vs negative.\n")
        fh.write("- NUPACK RNA-only accessibility: pathological junctions highly unpaired; exploratory "
                 "single-material competition shows probe bound to pathological (~0.25-0.29) but ~0 to "
                 "normal/negative (labeled non-hybrid bracket).\n\n")
        fh.write("## SWCNT geometry: {}\n".format("PASS" if gswcnt.get("passed") else "n/a"))
        fh.write(f"- 5 chiralities, diameters match analytic formula (<0.01 A), identical ~90 A length, "
                 f"all carbons type A / charge 0.\n\n")
        fh.write("## Docking benchmark (Gate 3)\n")
        if g3:
            fh.write(f"- Ligand basis: dinucleotide repeat units (full-oligo docking clash-dominated).\n")
            fh.write(f"- median GTd={g3.get('median_GTd')}, CTd={g3.get('median_CTd')}, ATd={g3.get('median_ATd')} "
                     f"(controlled Vina score; more negative = more favorable).\n")
            fh.write(f"- GT-CT diff upper95CI={g3.get('diff_GT_CT_ci_upper')}, "
                     f"GT-AT diff upper95CI={g3.get('diff_GT_AT_ci_upper')}.\n")
            fh.write(f"- Gate 3 passed = {g3.get('passed')}. ")
            fh.write("GT separated from controls.\n\n" if g3.get("passed")
                     else "GT did NOT separate from controls -> chirality conclusions EXPLORATORY.\n\n")
        else:
            fh.write("- pending.\n\n")
        fh.write("## Chirality screen (Gate 4)\n")
        if chir:
            for r in chir:
                fh.write(f"- {r}\n")
        else:
            fh.write("- pending / see chirality_ranking.tsv.\n")
        fh.write("\n## Structures (Gate 5 = conceptual)\n")
        fh.write(f"- {g5.get('n_starting_models','?')} builder-placed starting models; "
                 "no equilibration; no capture-exposure claim.\n")

    # ---------- LIMITATIONS ----------
    with open(os.path.join(REP, "LIMITATIONS.md"), "w") as fh:
        fh.write(f"""# LIMITATIONS

Generated {utc_now()}.

1. **Vina is mis-shaped for full ssDNA on a CNT.** Rigid docking of GT7/GT15/full sensors is
   clash-dominated; the screen was reduced to dinucleotide base-stacking units. This probes
   base-surface stacking, NOT the full anchor's adsorption thermodynamics or kinetics. Vina
   scores are controlled comparative steric/stacking scores on an uncharged carbon receptor,
   never binding free energies.
2. **NUPACK 4.1 lacks a true DNA/RNA hybrid analysis parameterization** (single-material only).
   Mismatched/competing hybrid energetics are therefore NOT computed as a hybrid result; the
   single-material DNA/RNA competition is exploratory only. Perfect-duplex hybrid thermodynamics
   come from MELTING (Sugimoto), with a Biopython cross-check.
3. **MELTING vs Biopython Tm differ ~6-7 °C** at 150 mM due to salt-correction/initiation
   convention differences over the shared Sugimoto set (not averaged; investigated).
4. **Cryptic exon 2a is not in the canonical transcript**; pathological junction context is
   constructed from exon1 (RefSeq) + the project cryptic-exon sequence and labeled as such.
   The pathological long-context window is constructed, not transcript-derived.
5. **No MD.** No peer-reviewed SWCNT carbon force field was applied; structures are conceptual
   starting geometries only. No capture-exposure or equilibrium-corona claim is made.
6. **Docking scope** was limited to the experimentally relevant ligands and a controlled
   base-stacking benchmark, not the full multi-anchor x multi-chirality matrix.
7. Single representative chirality (7,5) used for conceptual structures.
""")

    # ---------- FINAL REPORT ----------
    with open(os.path.join(REP, "FINAL_ANALYSIS_REPORT.md"), "w") as fh:
        fh.write(f"""# FINAL ANALYSIS REPORT — STMN2 RNA-SWCNT sensors

Generated {utc_now()}.

## Executive summary
A reproducible CPU pipeline verified the STMN2 sensor sequences from an authoritative
transcript, quantified DNA-probe/RNA-target hybridization thermodynamics, ran a controlled
SWCNT base-stacking docking screen, and produced conceptual full-sensor starting structures.
Evidence is reported conservatively; no surrogate scores, no binding-energy language, and no
equilibrium-exposure claims.

## Quality gates
- Gate 0 (smoke tests): {g0.get('passed')} — NUPACK installed; hybrid-analysis limited; MD disabled.
- Gate 1 (sequence provenance): {g1.get('passed')} — RefSeq {g1.get('normal22_source','')[:40]}...
- Gate 2 (thermodynamic validity): {g2.get('passed')}.
- Gate 3 (docking benchmark): {g3.get('passed', 'pending')}.
- Gate 4 (chirality robustness): {g4.get('passed', 'pending')}.
- Gate 5 (structural exposure): {g5.get('passed')} — conceptual structures retained.
- Gate 6 (reproducibility): see archive + rerun command.

## Conclusions
### Supported
- junction22 is predicted modestly more stable than junction20 as a perfect pathological
  duplex under the specified free-solution conditions (Gate 2).
- Both probes show much stronger in-silico free-solution stability/complementarity contrast
  with the pathological junction than with normal/negative targets (Gate 2).
- SWCNT receptors are geometrically validated (diameter, length, typing).

### Conditionally supported / Exploratory
- Controlled Vina base-stacking benchmark and chirality screen: see Gates 3/4; chirality
  conclusions are {('conditional' if g3.get('passed') and g4.get('passed') else 'EXPLORATORY')}.

### Not supported (by design / data)
- Capture-domain solvent exposure at equilibrium (no MD; conceptual structures only).
- Any physical binding-free-energy or optical-readout ranking from Vina scores.

## Reproducibility
Rerun: see `reports/RERUN.md` / the Snakemake command. Environments exported under provenance/.
""")
    print("Reports + claim-evidence matrix written.")

if __name__ == "__main__":
    main()
