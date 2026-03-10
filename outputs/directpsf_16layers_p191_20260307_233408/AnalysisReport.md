# directpsf_16layers_p191_20260307_233408 Result Analysis

## 1. Experiment Setup

- Run directory: `outputs/experiments/directpsf_16layers_p191_20260307_233408`
- Depth-dependent PSF mode: `DirectPSF`
- Depth interpolation: nearest two-layer linear interpolation
- QE: fixed (`fix_qe=true`)
- Epochs: `40`
- Batch size: `2`
- Depth range: `[0.3, 1.5] m`
- Focus distance metadata: `0.9 m`
- Number of depth layers: `16`
- PSF size: `191 x 191`
- PSF initialization: `delta`

## 2. Final Metrics

From `summary.json`:

- Best validation total loss: `0.11055`
- Best epoch: `33`
- Final training total loss: `0.06857`
- Final validation total loss: `0.11438`
- Final validation RGB loss: `0.03211`
- Final validation depth loss: `0.00251`

Overall judgment:

- Training converged normally.
- The model did learn a useful solution.
- The best checkpoint was reached before the last epoch, so evaluating `best.pt` rather than the last epoch is important.

## 3. Comparison Against the Earlier 8-layer / 127-PSF Run

Reference run:

- `outputs/experiments/directpsf_run_20260307_171242`
- `num_depth_layers=8`
- `psf_size=127`
- best validation total loss: `0.09547`

Comparison:

- `8 layers / 127`: best val `0.09547`
- `16 layers / 191`: best val `0.11055`

Interpretation:

- The larger model did **not** outperform the smaller baseline on validation loss.
- Increasing both `num_depth_layers` and `psf_size` increased optimization freedom and cost, but the extra freedom did not translate into a clear validation gain.
- At least for the current loss, data, and training recipe, the larger optics model appears harder to optimize and not more effective yet.

## 4. Speed and Cost Analysis

This run is much slower than the earlier `8 x 127` configuration.

Approximate throughput from `history.json`:

- Train throughput: about `2.21 it/s`
- Val throughput: about `5.63 it/s`

Compared with the earlier `8 x 127` run:

- Earlier train throughput was around `7.4 it/s`
- Earlier val throughput was around `18 it/s`

So the new configuration is roughly `3.3x` slower.

This slowdown is expected because:

- `num_depth_layers` doubled from `8` to `16`
- `psf_size` increased from `127` to `191`
- The spatial convolution cost grows roughly with kernel area

The speed penalty is therefore large, while validation performance did not improve correspondingly.

## 5. Loss Curve Interpretation

From `logs/loss_curve.png`:

- Training loss decreases smoothly over the whole run.
- Validation loss decreases overall, but still fluctuates.
- The validation curve improves substantially during early and middle training.
- After about epoch `30`, the gains become smaller and more unstable.
- Best validation performance appears at epoch `33`, after which the run no longer improves meaningfully.

This suggests:

- The system is trainable.
- The optimization remains somewhat noisy.
- More model capacity alone is not the bottleneck right now.

## 6. PSF Statistics Interpretation

From `logs/psf_curve.png` and `history.json`:

- `psf_peak_mean` drops from about `0.79` to about `0.68`
- `psf_center_3x3_mass_mean` drops from about `0.79` to about `0.70`
- `psf_entropy_mean` stays around `2.3 - 2.45`
- `psf_second_moment_mean` drops from about `1586` to about `1411`

Interpretation:

- The PSF bank did not remain purely delta-like.
- Some kernels spread out and became more structured.
- However, the overall bank still keeps a large amount of energy concentrated near the center.
- The statistics do not indicate a smooth, globally distributed blur family. Instead they suggest a mixed regime: some kernels remain very sharp, while some channels/layers become much more structured.

## 7. PSF Structure Analysis

Representative slices in `psf_representative_overview.png` show:

- Near depth (`depth 00`): all RGB channels remain almost delta-like.
- Mid depth (`depth 08`): the red channel changes strongly and becomes visibly irregular; blue changes locally; green remains near-delta.
- Far depth (`depth 15`): green shows a compact non-delta structure; red and blue remain close to central peaks.

Additional numerical checks on `psf_bank.npy`:

- `depth 0` and `depth 4`: RGB peaks still sit at the center with peak value about `0.9648`
- `depth 8`: red has a weak off-center maximum near `(101, 93)`, green remains almost perfectly centered, blue expands but stays centered
- `depth 12`: all channels are centered but broadened, especially blue and green
- `depth 15`: red and blue return to almost delta-like central peaks, while green remains broadened

This means:

- The learned PSF family is **not** smoothly evolving across all 16 depth layers.
- Instead, only certain depth/channel combinations are heavily used for coding.
- The optimizer is exploiting a sparse subset of optics degrees of freedom.

## 8. Why the PSF Still Looks Irregular

The irregularity is surprising from a classical optics viewpoint, but it is consistent with the current model design.

Reasons:

1. The current `DirectPSF` parameterization is fully free-form.
  It enforces only non-negativity and sum-to-one normalization.
2. There is no physical regularization.
  The PSF is not constrained to be:
  - radially smooth
  - centered
  - symmetric
  - explainable by a real phase element
3. The task is loss-driven, not optics-driven.
  The optimizer will use any PSF structure that helps RGB reconstruction and depth estimation, even if that structure looks non-physical.
4. More capacity makes this easier.
  With `16` layers and `191 x 191` kernels, the optimizer has much more freedom to create specialized task codes.

So the current result should be interpreted as:

- a valid task-optimized PSF bank
- but not yet a physically convincing optical design

## 9. Image Reconstruction Quality

Best saved samples show:

- Valid RGB regions are reconstructed reasonably well in many cases.
- Large masked regions still show hallucinated color fills, which is expected under the current masked loss design.
- Depth maps are globally plausible, but shape boundaries can still be overly smooth or shifted.

In particular:

- The model often captures the coarse scene layout correctly.
- The depth head tends to produce smoother structures than ground truth.
- RGB reconstruction inside valid regions is usable, but not yet sharply detailed.

## 10. Main Conclusion

This run proves that:

- a larger `16-layer`, `191 x 191` PSF bank can be trained successfully
- the PSF bank does change meaningfully
- some depth/channel slices are actively used as coding optics

However, the run also shows that:

- the larger optics parameterization is much more expensive
- validation quality is not better than the smaller `8-layer / 127` baseline
- the learned PSFs remain irregular and only partially structured
- more optical freedom alone is not enough to guarantee a better result

## 11. Recommendation for Next Steps

Based on this run, the most reasonable next steps are:

1. Do not scale DirectPSF much further yet.
  Increasing both depth layers and kernel size again is unlikely to be efficient before improving the model prior.
2. Keep using the logging system and best-checkpoint export.
  This run confirms those tools are necessary.
3. Move toward phase-mask-based optimization.
  The current DirectPSF results are useful as an upper-flexibility reference, but they are already showing the limits of unconstrained free-form kernels.
4. If DirectPSF is revisited later, add mild structure priors.
  Good options would be:
  - center bias
  - smoothness prior
  - inter-layer continuity prior
  - low-rank or basis parameterization

## 12. Files Worth Inspecting

- `summary.json`
- `logs/history.json`
- `logs/loss_curve.png`
- `logs/psf_curve.png`
- `psf/psf_representative_overview.png`
- `psf/psf_bank.npy`
- `im/best_epoch_033_sample_00.png`
- `im/best_epoch_033_sample_01.png`

