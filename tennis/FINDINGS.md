# Tennis experiments — findings log

All results walk-forward by year, evaluated against de-vigged Pinnacle
closing odds (tennis-data.co.uk), features built chronologically from
Sackmann ATP data (main tour + qualifying/challengers, 181K matches,
2000-2018). 34,419 matches matched across sources (surname+initial pairs,
±10-day tolerance; market favorites win 70.1% — sanity check passed).

## Headline (21,324 matches, 2010-2018)

| model                              | log loss |
|------------------------------------|----------|
| Pinnacle closing (de-vigged)       | 0.5611   |
| Elo/form/H2H (v1)                  | 0.5795   |
| + serve/return features            | 0.5781   |
| points-share (dominance) target    | 0.6186   |
| blend (rich + dominance)           | 0.5890   |
| walk-forward stack (market+blend)  | 0.5613   |

## Findings

1. **Serve/return features genuinely help** — beat the Elo-only model in
   8/9 years. More/better data does improve the model *relative to
   itself* (~0.0014 LL). It closes <10% of the gap to the market.

2. **The NCAAB margin trick does NOT transfer to tennis.** Points-share
   regression (basketball: better in 21/21 seasons) is far worse here
   (0.6186). Cause: tennis scoring has clutch structure — 2.9% of matches
   are won on a minority of points, and that skew correlates with player
   quality (big-point performance). Raw point share systematically
   underrates clutch winners. Technique transfer requires the target to
   mean the same thing in the new sport.

3. **First nonzero stack weight of the project**: 0.19 on the model
   (NCAAB: 0.01). But the payoff is 0.0001 LL — a trace of orthogonal
   information, economically nil. Verdict unchanged: no edge vs closing.

4. **De-vig sharpening is source-dependent.** Optimal market-logit
   weight: Pinnacle 0.96 (≈ none needed), NCAAB multi-book consensus
   1.14. Low-vig sharp books are already calibrated; average-of-books
   consensus is shrunk toward 50% and should be sharpened ~1.1x.
   => Fair-value engine rule: prefer Pinnacle-grade sources raw;
   sharpen blended consensus.

## Replication notes

- Elo: FiveThirtyEight decaying K (250/(n+5)^0.4), 50/50 overall/surface
  blend, updated over ALL tiers (challengers improve young-player ratings).
- Rolling windows: last 20 matches for serve stats, 10 for form.
- Time-decay sample weights (3-year half-life) in training.
- Dominance mapping: predicted share -> win prob via 1-D logistic fit on
  the inner (early-stop) year only.
- Stack: logistic on (market logit, model logit), fit strictly on prior
  years' validation predictions.
