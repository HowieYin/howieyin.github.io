# Two content review passes — 2026-10-27

## Pass 1: completeness, source fidelity and mechanism reconstruction

Checked PDF, arXiv HTML and TeX for Abstract, §1–§6, Acknowledgements/Impact and A.1–A.5.
All5figures preserved uncropped;13table matrices compared cell-by-cell including spans and bold/italic/underline.
All4numbered equations and A.3 formulas retained. Added formulas labeled Interpretation.
Verified anchor+B−1 masks, τ includes bonus, causal target verifier, bidirectional block discard, training context stopping before anchor.
Checked snapshot94e4abc source path, target layer offset/config override, deterministic draft vs sampled target, crop/update lifecycle.
Fixed duplicate SVG ids; source table manifest anchors repaired from untouched paper HTML.

## Pass 2: numerical claims, references and evidence limits

RecomputedTable1/3/4/5/6/7/8/9/11/12/13 comparisons.
Preserved and called out Figure1caption >2.5× vs EAGLE60 average2.406×, §5.1wrongEAGLE16label, no-featureTable10valuesabove3×.
Separated τ gains fromspeedup; longcontextTable4doesnotproveendtoendspeed.
Recomputed A.3BF16 projection40MiB, activation40/8MiB, B16transient384KiB.
Added persistentKVscaling with explicitly hypothetical5L/8KV config; no total-memory42MB claim.
Flagged TTFT andfirst-draftprefill omission in officialreturn_stats without claimingpaperusesexacttimingpath.
Loss index toy convention explicit; reduction/anchor-index implementation notinvented.
20externalURLs returnedHTTP200, citationtitles corrected toofficialmetadata.
Updated related papers through2026-10-27, keptDFlash2blogdistinctfrompaper v2, DLoopsame-studyincrementandDHiSparseconditionalcomparison.
No localH200/B200 model reproduction; benchmarks attributedtopaper/externalauthors.

Supplemental official-source review: Z Lab / Modal / SGLang June15 integration article returnedHTTP200; immediateKVmaterialization,layer-batchedprojection,norm/RoPEfusion andSpecV2hostoverlap distinguished frommodelarchitecture gains. ExamplelaunchB8doesnotmatchbenchmarkB16.
ConcurrentEAGLE-2andMedusareadings retained inarchive andlinked inbothdirections.
Browser/rendering QA evidence is recorded separately in local and production reports.
