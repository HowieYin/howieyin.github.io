# Source-grounded research findings

Read all 14 pages of supplied 1607.06450v1.pdf; PDF metadata created 2016-07-22; arXiv submission 2016-07-21. No conference claim verified: label arXiv preprint. Three authors: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (Toronto; Hinton also Google).

Original order:
Abstract → 1 Introduction → 2 Background → 3 Layer normalization → 3.1 RNN → 4 Related work → 5 Analysis → 5.1 Invariance → 5.2 Geometry → 5.2.1 Riemannian metric → 5.2.2 normalized GLM → 6 Experiments (6.1 OE, 6.2 attentive reader, 6.3 skip-thought, 6.4 DRAW, 6.5 handwriting, 6.6 permutation-invariant MNIST, 6.7 CNN) → 7 Conclusion → References → Supplementary (LN definition, LSTM, GRU, DRAW, gain geometry).

6 figures, 3 tables, numbered equations 1–35. Full sources in sources/paper.tex, paper.bbl, equations.json and table-*.html. Equation 6 is multirow; preserve preceding Ex1 rows. Equations 1–4 appear twice in arXiv HTML responsive variants, deduplicate. Primary visuals extracted in assets/original with full captions; original source figure PDFs also retained. Figure 4 comes after Figure 5 in PDF due float positioning; explain in conceptual original subsection order.

Mechanism:
a = W h (H-vector), normalize across H hidden units on one sample, μ=mean a, σ=sqrt(mean((a−μ)^2)), then gain g and bias b before f. LN g/b H-vectors, stats shared within sample but independent across batch. Modern Transformer B,T,D: reduce D independently for each b,t, stats B,T,1; don't reduce B or T (causal leakage). Population variance 1/D, correction=0. Per feature affine vs shared stats. g/b output not zero-mean/unit-var in general. No new task loss, no KV cache, no running stats. RNN current state dependence doesn't prove arbitrary-length generalization or absence of gradient explosion. Cost O(BTD); dense projections O(BTD²); small ops still reduction/launch/IO costs.

Math:
1 linear projection; 2 BN expectation data distribution; 3 LN H statistics; 4 vanilla RNN normalized preactivation; 5 shared formulation LN/BN/WN; 6 W'=δW+1γᵀ positive δ equivalence because common shift γᵀx; 7 per-sample positive x rescaling; 8 KL quadratic 1/2 δᵀFδ; 9 Fisher expectation model y distribution; 10 exponential-family GLM logP; 11 mean/variance; 12 Fisher E[Cov(y|x)/φ² ⊗ augmented xxᵀ]; 13 normalized Fisher block using [w_i,b_i,g_i], dimensions (d+2)^2, i,j blocks; 14 χ_i=x−∂μ_i/∂w_i−(a_i−μ_i)/σ_i ∂σ_i/∂w_i, vector d; 15–16 LN function; 17–19 standard LSTM; 20–22 normalized LSTM; 23–25 standard GRU; 26–28 normalized GRU; 29–31 DRAW only cell output LN; 32–35 gain-direction Fisher geometry.

Useful agent derivation:
P=I−11ᵀ/D; r=sqrt(v+eps); z=(a−μ1)/r.
Jacobian J=(P−zzᵀ/D)/r, dy/da=diag(g)J.
J 1=0 exact; ε=0 J z=0, J rank D−2 for nonzero variance; eps>0 J z=(eps/r³)z, radial effect persists. D=1 all input gone (output beta); D=2 standardized vector only sign direction (eps=0).
Backprop q=upstream*g: dx=r^-1 [q−mean q−z mean(qz)], dgamma=sum(upstream*z) over samples/tokens, dbeta=sum upstream. Correct eps version too.
Toy a=[1,2,3,4], μ=2.5 v=1.25 z=[−1.341641,−.447214,.447214,1.341641].
With δ>0 and eps=0 LN(δa+c1)=LN(a); δ<0 normalized vector flips sign; general feature shifts aren't common scalar shifts. eps>0 destroys exact scale invariance, common offset still exact mathematically. Large common offsets in finite precision may erase relative differences.

Critical source discrepancies (must clearly distinguish preserved original from corrected interpretation):
Table1 BN [Yes,No,Yes,Yes,Yes,No], WN [Yes,No,Yes,No,No,No], LN [Yes,Yes,No,Yes,No,Yes] for whole W scale, common-row W shift, individual row scale, dataset scale, dataset shift, one-sample scale.
Body says all methods dataset-scale-invariant; WN counterexample g wᵀ(δx)/||w|| scales δ: table No correct. Conclusion says per-training-case feature shifting; applies to activation a common offset, not arbitrary raw x+q (Wq generally nonconstant). Positive scales assumed implicitly. Bias must be after LN or transform jointly.
Appendix Eq15 α=gain β=bias but prose reversed and says α zeros β ones. Follow formula, body default gain1 bias0 and original author implementation s=1,b=0. Preserve original formula and explain typo explicitly. Original code ln uses epsilon1e-5 and float32.
GLM Fisher analysis doesn't establish full-network optimizer/convergence theorem, and "early stopping" is analogy. Joint scaling W gives σ~δ, weight-weight Fisher factor 1/δ² (not 1/δ); σ doesn't grow from independently scaling one row under LN. BN Eq32 simplifies standardized factors inside expectation too aggressively: z has unit marginal variance but E[Cov*z²] needn't equal E[Cov]; identify dependency assumption.
Eq35 PDF also appears to have a ||w_i|| denominator where j expected; keep original and use corrected symmetric expression in explanation if addressing.
