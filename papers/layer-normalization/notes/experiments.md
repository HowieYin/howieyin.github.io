# Experiments and exact results

§6 defaults gains1/bias0 except noted.

6.1 OE: MSCOCO; sentence GRU and pretrained VGG image 10-crop embeddings; Theano, Adam same initial hyperparams and architecture; validate every300 iterations; choose best R@K; test mean5 splits, each1000 images+5000 captions. Figure1 panels image R1/R5/R10 vs iteration×300; LN blue vs baseline green; fewer updates / authors say reaches best in60% baseline time, cannot derive universally60% wall time. Table2 all8 metric columns plus model:
Sym: caption45.4, blank,88.7,5.8; image36.3,blank,85.8,9.0
OE Vendrov:46.7,blank,88.9,5.7;37.9,blank,85.9,8.1
OE ours:46.6,79.3,89.1,5.2;37.8,73.6,85.7,7.9
OE+LN:48.5,80.6,89.8,5.1;38.9,74.3,86.3,7.6
Baseline ours same implementation best causal comparator. CaptionR1+1.9pp/+4.08% relative; imageR1+1.1pp/+2.91%; mean rank decreases .1 and .3. No hallucinated R5 missing entries. Historical SOTA for RNN embeddings not all retrieval; Wang different test single split not comparable.

6.2 attentive reader CNN corpus anonymized entities and truncated4sentences; uni-LSTM; baseline and two BN variants (LSTM-only/everywhere) from Cooijmans curves; LN only LSTM. Figure2 x trainingsteps thousands, y validation error. LN cyan faster/better vs blue LSTM, green BN-LSTM, red BN-everywhere; visually ~.49 but avoid exact digitized final. Validation only (not test); scale1 better than.1 does NOT prove universal insensitivity. Official author README command dim240/batch64/lr8e-5/Adam, separate code settings from paper, README warns repo settings changed. Hardware/precision not full protocol.

6.3 skip-thought: BookCorpus encoder/decoder GRU, encoder dim2400; same hyperparams, Theano; CNMeM no significant perstep difference reported (not measured here); checkpoints50k until1M. Figure3 six panels SICK Pearson×100 up/MSE×100 down/MR/CR/SUBJ/MPQA accuracy up; green ours vs blue LN; gray Original; shaded error10fold classifier CV not multiple training seeds.
Table3 Original .848,.778,.287,75.5,79.3,92.1,86.9
Ours .842,.767,.298,77.3,81.8,92.6,87.9
LN .854,.785,.277,79.5,82.6,93.4,89.0
LN† .858,.788,.270,79.4,83.1,93.7,89.3
†1month~1.7M vs others1M; bold best except MR79.5 in1M LN. Samebudget SICK MSE-.021/−7.05%, MR+2.2pp/+2.85%; CR+.8pp SUBJ+.8 MPQA+1.1.
Original/publiccode mismatch: published original conditioned decoder every timestep; public implementation not, authors say possible reason original vs ours differ. Don't compare original changes to LN alone.

6.4 DRAW: fixed binarization Larochelle/Murray, 50k/10k/10k; 64glimpses 256LSTM, Adam defaults,batch128; Fig4 baselineblue WNgreen LNred,testvariational bound vs epoch0–100; after200epochs baseline82.36nats LN82.09: -.27 nats/-.328%. Variational bound not exact NLL. Almost2xfaster convergence claim not kernel2x; gains near final small. LN only ct before tanh, not all gates (Appendix29–31).

6.5 handwriting: IAM-OnDB 221writers12,179lines; meanlength~700, input string>25characters; three400LSTM layers, total~3.7M, output20 bivariate Gaussian components, input3, char/window57, window10Gaussians×3params=30; Adam batch8; Fig5 trained subsequenceslength500, x updates×200 logarithmic, y negative NLL (can be negative continuous density, not discrete prob contradiction); four curves train/test baseline/LN; reaches comparable final much faster; no new sample visual, no Table metric final.

6.6 MLP permutation-invariantMNIST (not sequential/pixelRNN):784–1000–1000–10,55k train,Adam,batch128 vs4; LN hiddenlayers only excluding softmax; BN all layers baseline fairness concern outputnorm discrepancy and smallbatch BN uses unbiased variance; Fig6 trainNLL log top, testerror bottom; left128 right4; benefit strongest convergence with small batches, all test~1% order, no precise new scores. Batch robust normalization != same SGD generalization across batch.
6.7 CNN preliminary qualitative only BN beats LN/baseline; hypothesized spatial boundary/inactive units distribution differences, not proven causal controlled ablation. No omitted CNN numerical table or Figure.

Audit gaps: GPU,precision,wallclock tables,throughput/p95,fullparams/trainingtokens/B/seq for some tasks, seedvariance; don't invent. No systematic mean-only/variance-only/gain/bias/LN-location/optimizer factorial ablation. Six tasks not six controlled independent universality proofs.
