# Future directions — LLM-MovieLens follow-up work

**Role of this folder.** The single canonical home for all post-CIKM / post-NeurIPS follow-up planning, working evidence, and isolated experiments that build on the released LLM-MovieLens artifact. This is *not* a submission — it is the staging ground for the next 1–3 papers. **This README is the one place for the future directions** (the former `docs/NEXT_PAPER_STRATEGY.md` has been merged in).

**Status:** Active planning. The 4-method × 4-density matrix work (R2/R3 on subsampled-ML-20M) has been **promoted into the CIKM/NeurIPS submissions** — those papers now report the replacer class triangulated across all four datapoints (R2 −5.1%, R3 −3.9% vs M7 at the sub163 control). All sub163 artifacts (winner checkpoints + grid-sweep result JSONs) were relocated into the neurips tree (2026-05-25); this folder now holds only the broader follow-up directions + the reusable retune harness. Awaiting a decision on which next paper to launch.

---

## Salami-slicing constraint (read first)

A top-tier next paper MUST be substantively distinct from the CIKM/NeurIPS submission or it is desk-rejected as salami-slicing (NeurIPS and CIKM both check). **"Same density-paradigm framework + 2 more matrix cells" does NOT clear that bar** — the CIKM paper already monetises the *descriptive law*. The next paper needs a new axis: a new method (①), new domains (②), new task (③), new architecture (④), or a bridging benchmark (⑤). It must cite the CIKM paper as prior work and frame itself as "law → generalization + method," never "more of the same."

**Litmus test:** if a reviewer can substitute the CIKM abstract for the next paper's abstract, it dies. The ①+② framing passes; pure ② alone is borderline; "+2 matrix cells" fails outright.

---

## The asset in hand (completed 4×4 matrix)

A density-stratified, two-instantiation-triangulated characterization of LLM-for-RecSys paradigms — on **one domain family** (ML-20M variants + a single Amazon-Books point). Completed 4×4 (Δ NDCG@10 vs M7):

| Method | ML-20M (1,160) | sub163 (225) | ML-1M (163) | Amazon (13) |
|---|---|---|---|---|
| R1-gene (regularizer) | −1.1% | +3.7% | +17.0% | +29.5% |
| R1-plus (regularizer) | −5.6% | +10.9% | +16.3% | +26.4% |
| R2 (replacer) | −2.6% n.s. | −5.1% | ties | −22.0% |
| R3 (replacer) | −6.5% | −3.9% | ties | −11.7% |

> ML-20M cells are the paper-canonical *d* = 128 capacity-matched values; matches CIKM/NeurIPS Table 4 + density chain.

**Framing discipline (carry forward):** regularizers gain monotonically with sparsity; replacers are envelope-bounded by injection (never beat M7) but are **NOT** monotone — the chain (−2.6 → −5.1 → tie → −22) falsifies replacer monotonicity, so never claim it.

---

## Strategic framing — independent rubric score (8.5/10, reviews excluded)

An independent `top-venue-paper-craft` scoring of the submitted CIKM paper (paperreview.ai set aside) placed it at **8.5/10** (fair-reviewer range 8–9): top-decile experimental rigor (all 7 robustness axes + two-instantiation triangulation), best-paper-tier reproducibility (53-cell verifier, Croissant, datasheet, plug-in), exemplary statistical honesty. **It is Best-Resource-Paper-competitive today.** The gap to a CIKM *Full-Research* best paper is three specific things, each mapping to a direction:

| Limiter (from the score) | Direction that closes it |
|---|---|
| **Descriptive, not prescriptive** — the density law is characterized; no method exploits it | **①** density-adaptive selector (turns the law into a method with a gain — single highest-leverage move) |
| **Modest within-domain effect (~3%) + 4-datapoint / 2-domain evidence base** | **②** cross-domain replication (widens the evidence base; the +26–30% sparse swing becomes a multi-domain law) |
| **Dense prose / number-heavy abstract** | a craft pass on the next paper (not a research direction) |

**Strategic read:**
- **Full-Research best-paper run → ①+② together** (method + wide evidence). ① alone risks the "trivial threshold" critique; ② alone is borderline salami. Combined, they convert all three limiters at once.
- **Resource / Reproducibility best-paper run → ⑤** (bridging-protocol benchmark), which leans on the already-Best-Resource-competitive strength without a new method.
- ③ (mood) and ④ (sequential) are independent methods papers, parallelizable. The "predictive theory of *why* density determines paradigm" is **a mechanism section folded into ①**, never a standalone paper (must survive a theory reviewer).

Recent CIKM precedent: Full-Research best papers carry a method-with-a-gain or a striking finding (CIKM'24 *Physics-guided Active Sample Reweighting*; *Data Void Exploits*); resource/benchmark artifacts win the Resource track (CIKM'25 *Semantic IDs in Generative Recommendation: A Practitioner's Handbook*).

---

## Future directions (5 candidates, ranked)

> **Reviewer-validated (paperreview.ai, 2× ACCEPT on the CIKM submission).** Both reviews endorse the density-adaptive switcher (①) and broader-domain replication (②), ask for stronger sequential baselines (④; BERT4Rec/DuoRec), and back the mood user study (③); their single biggest gap — no side-by-side vs topology-/token-level integration — seeds the new ⑤.

| # | Direction | Profile | Venue ceiling | Effort |
|---|---|---|---|---|
| **①** | **Density-adaptive paradigm-switcher** — a selector that, given a target catalog's per-item density, routes/blends regularizer vs. injection vs. replacer and beats every fixed paradigm across the density spectrum. Turns the framework descriptive → **generative**. | New mechanism (best-paper Profile A) | NeurIPS main / KDD / SIGIR / WWW | Medium |
| **②** | **Cross-domain replication of the density law** (4-method × ≥3-density × 5-seed matrix on Yelp/Steam/MIND/Goodreads). Tests whether the monotonicity + replacer collapse are domain-invariant. | Wide blast radius (best-paper Profile B) | NeurIPS D&B / KDD D&B / SIGIR Resource / CIKM | Medium-high |
| **③** | **MoodSteer — controllable retrieval via named mood axes** — formalise zero-shot `q + α·e_axis` into a method; user study + cost comparison vs learned re-rankers / free-form steering. | Methods (separate from density framework) | RecSys / SIGIR / WWW / UMAP | Medium |
| **④** | **Sequential × content × density** — proper M7-into-sequential fusion (input/FFN/output injection points), with **BERT4Rec + DuoRec** strengthening the dense endpoint. | Methods (sequential axis) | RecSys / WSDM / CIKM / KDD | High |
| **⑤** | **Bridging-protocol density benchmark** (NEW; the reviews' biggest gap) — bridge protocols so feature-injection (ours) vs CF→token-mapping vs topology vs graph-diffusion are comparable on one testbed, ranked *across density*. | Benchmark / evidence | SIGIR (repro/resource) / CIKM / RecSys (repro) / KDD D&B | Medium-high |

**Recommendation: ①+② together** — combines a new method (selector) with a new evidence base (domains the framework hasn't seen); cleanest top-tier story and strongest salami defense ("CIKM established the law on movies+books; this paper proves it domain-invariant AND turns it into a zero-shot selector that beats every fixed paradigm on held-out domains"). **⑤** is the strongest *evidence-only* fallback if a method isn't ready. ③/④ run in parallel without touching the framework story.

> **Make it predictive, not post-hoc** (lower-priority lever, from `project_llmmovielens_best_paper_directions.md` §3): pre-register the density→paradigm prediction on a held-out domain *before* running, then confirm. Converts the pre-spec-vs-post-hoc honesty asset into a headline. Pairs naturally with ②.

---

## Elaborated experiment plans (per direction)

Deep, runnable plans. Each is self-contained: hypothesis → approach → experimental design → expected outcome + falsifiers → effort/risk → venues + peer-reviewed anchors (skill Invariant 13). The reusable retune harness (`scripts/run_r{2,3}_retune{,_sweep}.py`) + the isolation pattern (`docs/ISOLATION.md`) underpin all of them. The protocol (full-ranking, temporal split, 5 seeds, paired-$t$, cold/medium/warm stratification) carries over verbatim from the CIKM paper.

### ① Density-adaptive paradigm selector — *headline method; converts descriptive → prescriptive*

**Hypothesis.** The density law is *exploitable*: a selector that reads catalog / per-item density statistics and routes each item (or dataset) to the predicted-winning paradigm beats every FIXED paradigm across the full density range, zero-shot on held-out domains.

**Approach (two selectors, escalating novelty).**
- *(a) Interpretable threshold policy* — per-item rule: density above $\tau_\text{high}$ → injection (M7); below $\tau_\text{low}$ → regularizer (R1-gene); blend in the band. Fit $\tau$ on the 4-datapoint chain. The "obvious from the law" baseline — include it precisely so the learned model must beat it.
- *(b) Learned meta-router* — features per item/catalog: per-item density, per-user density, degree-distribution Gini, catalog size, cold-bucket fraction, a content-SNR proxy (profile-vs-genome NN-overlap). Target: best paradigm (classification) or soft blend weight (regression). Train on the matrix cells (+ ②'s domains); evaluate **zero-shot** on entirely held-out domains.
- *Per-item routing* is the novel core: one model applies injection to warm items and regularizer to cold items **within the same catalog** — a density-conditioned mixture no fixed paradigm can match.

**Experimental design.** Train on the 4-density matrix + ②'s domains, hold out ≥2 domains entirely. Baselines: each fixed paradigm (injection/regularizer/replacer-always), **oracle** (best-per-cell upper bound), a **density-feature-ablated** router (must collapse toward a fixed paradigm), random router. Metrics: NDCG@10/Recall@10/MRR per domain; **regret vs oracle**; **win-rate vs best-fixed-paradigm on held-out domains**; per-bucket lift. Ablations: per-item vs per-dataset routing; threshold vs learned; feature importance; $\tau$ sensitivity. 5 seeds, paired-$t$.

**Expected outcome + falsifiers.** Claim: selector ≥ best-fixed-paradigm on EVERY held-out domain, zero-shot, no oracle. *Falsifier:* cannot beat best-fixed on held-out (→ law not exploitable; still publishable as a negative result backed by ②). *Trivial-threshold risk:* mitigate by showing per-item > per-dataset routing and learned > threshold.

**Effort / risk.** Medium (selector is light; ②'s matrices are the cost). Risk = "obvious from the law" → mitigated by zero-shot held-out dominance + per-item routing. **Venues:** NeurIPS main / KDD / SIGIR / WWW; journal ACM TORS / TKDE. **Anchors (peer-reviewed):** RLMRec, KAR, LightGCN.

### ② Cross-domain invariance of the density law — *widens the evidence base*

**Hypothesis.** The regularizer-vs-injection monotonicity + replacer collapse are properties of CF-graph density, not of any domain — they replicate across N new domains.

**Approach.** Rebuild the **4-method × ≥3-density × 5-seed** matrix on 3–4 new domains, each subsampled to ≥3 within-domain density points (like the sub163 control) so each domain yields a density *curve*, not one point. Domains for spread + diversity: **Yelp** (services, sparse), **Steam** (games, mid), **MIND** (news, extreme sparsity + temporal churn), **Goodreads** (denser books, complements Amazon-Books). Profiles via the released prompt template, domain-adapted; ~\$14–20/domain.

**Experimental design.** Per-domain: 4×4 matrix; **Spearman** of regularizer-gap vs $1/\text{density}$ (monotonicity); replacer-collapse check. Pooled: regression `gap ~ 1/density + domain-fixed-effects` — is density the dominant predictor, domain second-order? **Pre-register** the per-domain refuting result before running — converts the honesty asset into a headline (a *confirmed pre-registered* cross-domain prediction).

**Expected outcome + falsifiers.** Density explains the paradigm-gap sign/magnitude across domains; domain is second-order. *Falsifier (pre-registered):* a domain with non-monotone regularizer gap, or where a replacer wins.

**Effort / risk.** Medium-high (per-domain matrix is the cost center; reuse `run_r{2,3}_retune*.py` via the dataset-branch recipe). Risk: a domain breaks the law → publishable either way. **Venues:** NeurIPS D&B / KDD D&B / SIGIR Resource→full / CIKM full / RecSys; journal ACM TOIS / TORS. **Anchors:** domain dataset papers + RLMRec / KAR. Earlier scoping: `../neurips-2026-ed-track-llmmovielens/docs/future-cross-domain-extension.md`.

### ③ MoodSteer — controllable retrieval via named mood axes — *independent methods paper; needs a user study*

**Hypothesis.** Named-axis mood conditioning (`q + α·e_axis`) yields controllable retrieval users perceive as more mood-matched than profile-only similarity, at lower cost than learned re-rankers or free-form text steering, with acceptable relevance trade-off.

**Approach.** Formalize the steering operator + per-axis knob $\alpha$; restrict to the **4 HIGH-stability axes** ($r\geq0.7$, deployment-safe). Compare to: (a) profile-only cosine, (b) free-form text steering ("make it darker" + re-embed), (c) a learned mood re-ranker.

**Experimental design.** Offline: $\alpha$-sweep steering curves; controllability (centroid shift / requested offset; baseline ~0; current 30.8%); off-axis drift; relevance-preservation ($\Delta$NDCG under steering). **User study** (pre-registered, IRB): N≥30 (Prolific, ~\$500–1.5k), within-subjects; steered vs unsteered top-$k$; perceived relevance + mood-match + diversity (Likert); preference rate + agreement; 30-profile pilot first.

**Expected outcome + falsifiers.** Users prefer steered results on mood-match at small relevance cost; named-axis beats free-form steering on controllability-per-cost. *Falsifier:* no mood-match preference, or steering tanks relevance.

**Effort / risk.** Medium (1 wk offline + 1 wk user study + IRB). Risk: user-study logistics → pilot first. **Venues:** RecSys / SIGIR / UMAP / WWW; journal ACM TORS. Full scoping: `../neurips-2026-ed-track-llmmovielens/docs/future-methods-paper-mood.md`.

### ④ Sequential × content × density — *sequential axis; reviewer-requested stronger baselines*

**Hypothesis.** A proper M7-content-into-sequential fusion changes the sequential axis's steep density decay; and the "steepest decay" conclusion must be re-checked against stronger sequential baselines.

**Approach.** Design content fusion at three injection points — input-token embedding, FFN injection, output projection — and ablate the choice. Add **BERT4Rec** and **DuoRec** beyond SASRec at the dense endpoint.

**Experimental design.** Per-density (4 datapoints) sequential-vs-content; fusion-point ablation; does content fusion flatten the 6.8× decay (make sequential density-robust)? 5 seeds, paired-$t$.

**Expected outcome + falsifiers.** Content fusion narrows the sparse-density gap; fusion point matters. *Falsifier:* content fusion does not help sequential at sparse density → sequential is intrinsically density-limited.

**Effort / risk.** High (multiple fusion designs × ablations × seeds). **Venues:** RecSys / WSDM / CIKM / KDD; journal ACM TORS. **Anchors:** SASRec (ICDM'18), BERT4Rec (CIKM'19), DuoRec (WSDM'22). Deferral note: NeurIPS paper L1615.

### ⑤ Bridging-protocol density benchmark — *Resource-track best-paper play*

**Hypothesis.** Bridged onto one full-ranking testbed, the four content-integration FAMILIES rank predictably across density (injection density-robust; regularizer gains sparse; replacer loses; token-mapping ∼ injection).

**Approach.** Reimplement/adapt each family's representative onto the shared LightGCN-SF + full-ranking + temporal-split testbed. **Anchor on peer-reviewed:** FACE (NeurIPS'25, CF→token), injection (ours), RLMRec (regularizer), KAR (replacer). **Situate** preprints (TAGCF topology, TextGCN-MLP graph-diffusion) — reimplement-with-explicit-caveat or cite-only; never load-bearing anchors (Invariant 13).

**Experimental design.** Each family × 4 densities × 5 seeds; **protocol-bridging validation** (each reimplementation must reproduce its published numbers on its home dataset *before* bridging); cross-family density ranking; per-bucket cold-start. Document every bridging adaptation.

**Expected outcome + falsifiers.** Families rank by the density law; token-mapping tracks injection. *Falsifier:* a family's bridged behaviour contradicts its published claims → the bridge is unfair (document + fix or drop).

**Effort / risk.** Medium-high (faithful reimplementation + per-family sanity check is the cost). Risk: unfair bridge → mitigated by the published-number sanity check. **Venues:** SIGIR (repro/resource) / CIKM full / RecSys (repro) / KDD D&B; journal ACM TOIS / TORS.

### Camera-ready / artifact-v2 strengthening — *de-risks ① and ⑤*

The revision-tier items below are not papers — but the **no-aggregate ablation** and **residual-ID replacer** become clean analysis sections in ① and ⑤. Run once, reuse everywhere.

---

## Review signal (paperreview.ai — both reviewers recommend ACCEPT)

Two independent reviews of the CIKM submission both recommend acceptance and converge on the same gaps. (A) is camera-ready strengthening; (B) seeds the directions above.

### A. Camera-ready / artifact-v2 punch list (revision-tier), ranked by emphasis
1. **No-aggregate-statistics prompt ablation (popularity-leakage control) — the #1 ask in BOTH reviews.** Re-generate profiles with mean-rating / rating-count removed; re-run M4/M7 on ML-20M; show the LLM-over-genome/title gap survives. ~1 generation pass + 5-seed re-eval. **Do first** — closes the only soundness caveat.
2. **Residual-ID replacer variant** — add a low-dim ID pathway to R2/R3, re-test at sparse densities; disentangles "replacer paradigm" from "total ID removal."
3. **No-LLM rich-text encoder control** — encode plot+metadata (not just title) with the *same* sentence-transformer; isolates LLM *synthesis* from *text coverage*.
4. **Sub163 sampling spec** — document the exact subsampling policy (degree-distribution preservation) in an appendix.
5. Appendix-tier: **M6 (themes) diagnostics**, **uniform tuning budget + uniform d=128 across densities**, **implicit-threshold (≥4.0) / no-10-core sensitivity**.

### Citation-readiness of review-named methods (Invariant 13 — verified May 2026)
| Method | Status | Use in a next paper |
|---|---|---|
| FACE | NeurIPS 2025 ✓ | benchmark anchor (CF→token mapping) |
| LLMRec | WSDM 2024 (Oral) ✓ | benchmark anchor |
| LLM-CF | CIKM 2024 ✓ | benchmark anchor |
| CoLLM | IEEE TKDE 2025 ✓ | benchmark anchor |
| ColdLLM | WSDM 2025 ✓ | benchmark anchor |
| SASRec / BERT4Rec / DuoRec | ICDM'18 / CIKM'19 / WSDM'22 ✓ | sequential baselines |
| TAGCF (2602.21099) | arXiv preprint | situate-only; re-verify before benchmarking |
| TextGCN-MLP (2510.12461) | arXiv preprint | situate-only |
| RecLM (2412.19302) | arXiv preprint | situate-only |
| ColdRAG (2505.20773) | arXiv preprint | situate-only |

> Skill rule: benchmark/headline baselines must be peer-reviewed top-venue work; preprints are *situated* in related work with a why-not-benchmark line, never load-bearing. Re-check each preprint's status at write time — several (TAGCF, RecLM) may be accepted by then.

---

## Effort / venue summary

| Path | New work | Venue ceiling | Risk |
|---|---|---|---|
| **①+② (recommended)** | selector + 3–4 domain matrices | NeurIPS main / KDD / SIGIR | medium |
| ② only | 3–4 domain matrices | NeurIPS D&B / KDD D&B / CIKM | low |
| ⑤ (Resource play) | bridge 4 families × density | SIGIR / CIKM / RecSys / KDD D&B | medium-high |
| ③ (mood) | MoodSteer + user study | RecSys / SIGIR / UMAP | medium |
| ④ (sequential) | content×sequential fusion | RecSys / WSDM / CIKM | high |
| theory add-on | bias-variance / spectral derivation | (folds into ①) | high — never gate on it |

---

## Concrete next-experiment list

1. **Pick the headline.** Default: ①+② combined (Full-Research best-paper play); ⑤ if going Resource-track; ③/④ as parallel independent papers.
2. **For ②:** scope 3–4 new domains (Yelp / Steam / MIND / Goodreads), build a 4-method × ≥3-density × 5-seed matrix on each. Reuse `scripts/run_r{2,3}_retune*.py` via the dataset-branch recipe (data-dir / emb-dir branch per dataset).
3. **For ①:** start with [`Experiment_V1_Density_Threshold_Selector.md`](Experiment_V1_Density_Threshold_Selector.md): a 2-expert threshold/soft-blend selector (`M7` + `R1`/`R1-plus`) tuned on validation before moving to the learned meta-router; then show selector ≥ best-fixed-paradigm on every held-out domain vs a no-oracle baseline.
4. **Pre-register a falsification** before running (e.g., "if R1 fails to beat M7 by ≥+10% on any domain at int/item < 50, the regularizer-monotonicity claim is falsified").
5. **First, run the two camera-ready items** (no-aggregate ablation; residual-ID replacer) — they de-risk both ① and ⑤ and become reusable analysis sections.
6. **Reuse the isolated-workspace pattern** (`docs/ISOLATION.md`) for every direction; the CIKM/NeurIPS submissions stay frozen.

---

## Assets in this workspace

- **`scripts/run_r{2,3}_retune{,_sweep}.py`** — the reusable per-domain retune harness (isolation pattern + dataset-branch recipe). Generic template for ②'s per-domain matrices (first used for the sub163 retune).
- **`docs/ISOLATION.md`** — the strict workspace-isolation protocol (CIKM never touched, NeurIPS read-only). Reuse for every direction.
- **`logs/`, `results/`** — grid-exploration logs + the non-promoted R2/R3 grid sweeps.
- **Completed & shipped:** R2/R3 sub163 cells are in the CIKM/NeurIPS papers; winner checkpoints at `../neurips-2026-ed-track-llmmovielens/code/benchmark/checkpoints_ml20m_sub163/{r2,r3}/`; provenance JSONs at `.../hparams/{r2,r3}/sub163_retune/`.
- **Memory:** `~/.claude/projects/-Users-nghiaduong-Desktop-bkai/memory/project_llmmovielens_best_paper_directions.md`.

---

## Cross-references + sync rule

- Parent index: [`../README.md`](../README.md) — workspace submission index. **Updating this README should sync the "Working dirs and follow-up projects" entry (and last-sync date) in the parent.**
- Frozen submissions (read-only source / never touch): [`../neurips-2026-ed-track-llmmovielens/`](../neurips-2026-ed-track-llmmovielens/), [`../cikm-2026-llmmovielens/`](../cikm-2026-llmmovielens/).
- d=32 legacy checkpoints (evidence reservoir): [`../r1-ml20m-old-hparams/`](../r1-ml20m-old-hparams/), [`../r1plus-ml20m-old-hparams/`](../r1plus-ml20m-old-hparams/).
