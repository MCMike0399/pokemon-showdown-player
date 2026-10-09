# VGC checkpoint reviews and experimental planning profiles

The continuous service reviewed exactly50 and100 new ladder games after its532-game baseline. The first cohort won26/50 and the full cohort49/100; the preceding100 won36/100. These are observational cohorts with different opponents, not a randomized ladder comparison. Both reviewed cohorts used the same selected strategic-v1 actor and corrected rain team.

The audits found all100 terminal postgame reports, no trajectory issues, complete submitted own sets/natures/stat-point allocations, and private requests/calculated own stats in all900 decisions. Opponent nature was disclosed in eight games; opposing allocations remained unknown. Champions stat points and IV behavior are described in [match collection](match-data-collection.md).

## Evidence behind the experimental changes

Sleep blocked26 actions in six games. Opponent setup executed30 times in24 games, including repeated Quiver Dance, Calm Mind with Speed Boost, and Shell Smash behind redirection followed by Water Spout. Sole-survivor Protect appeared25 times in20 games. These observations motivate forecasting denial, setup and the position created by a defensive turn; they do not prove an unplayed alternative would have won. Protect can serve recovery, residual, field-expiry, PP or variance purposes.

The original preview encoder rarely changed its preferred brought four when only the opposing six changed:4/100 cases in the sensitivity audit. That diagnostic measures responsiveness, not selection quality. More responsive probabilities under the new features did not demonstrate better preferred selections.

The opt-in profiles add:

- `strategic-mechanics-v2` and `strategic-v2`: exceptional-stat and move-power estimates, accuracy stages, targeted sleep forecasts with visible immunities, and a narrow sole-survivor Protect cost that preserves useful stalls.
- `strategic-v3`: public opponent types/base stats and own-set coverage, exposure, estimated speed and team-plan features. A prospective plan permits at most one Mega Evolution.
- `strategic-v4`: bounded setup/recovery reply hypotheses, approximate next-turn pressure/defense/initiative utility, and HP-dependent Water Spout/Eruption/Dragon Energy power at attack execution.

Damage, opposing speeds and continuation values remain estimates. They use own exact sets and observable opponent evidence, not hidden spreads or private opponent requests. The default encoder dimensions remain fixed, and feature profiles identify distinct collecting semantics. Historical arrays stay immutable; new-profile PPO uses fresh sampled own trajectories.

The two [Wolfe thought-process studies](vgc-strategic-research.md) and [second-video study](vgc-wolfe-second-video.md) inform threat ordering, team plans and the purpose of a turn. They are not fabricated expert-action labels. A frozen behavior corpus from all632 terminal games contains5,987 observed opponent move executions; fainting, status and selection censor those observations.

## Completed development results

Each row compares policies on complete matching seeds, sides, opponent policies, teams and frozen inference inputs. Different rows use different panels; raw totals are not comparable between experiments.

| Candidate | Incumbent wins | Candidate wins | Cases | Paired gains/losses | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Learned v2 |60|59|96|7/8|0.69638|Failed|
| Learned v3, turn temperature0.7 |57|52|96|9/14|0.89498|Failed|
| V3 at turn temperature0.25 |67|69|96|14/12|0.42251|Failed|
| V3 at turn temperature0.4 |67|66|96|13/14|0.64945|Failed|
| Learned v4 with pressure curriculum |61|68|96|19/12|0.14052|Failed|
| Frozen earlier v4 on the larger new panel |121|131|192|30/20|0.10132|Failed|
| Mega-focused control on larger panel |121|105|192|21/37|0.98763|Failed|
| V4 after four additional fresh updates |121|140|192|38/19|0.00817|Failed minimum gain|
| Fresh v4 confirmation, prospective5-point gate |136|139|200|27/24|0.38988|Failed|

The v4 gain was7.29 percentage points: promising but below the declared10-point minimum and insufficient under the one-sided paired p<=0.05 requirement. Rejected weights/reports remain retained locally; all of these independent final panels stayed unopened. The collecting live turn temperature is0.25. The v3 temperature0.7 experiment increased exploration relative to it.

One further experiment was declared before its new results: four128-game fresh on-policy updates,192 development cases and200 independent final cases, with fixed source, team pool, temperatures and promotion criteria. Training completed on512 unique episodes and4,183 decisions. All four updates ran on native MPS; collecting likelihood errors were below0.0001 and the trajectory audit was clean. The third and fourth updates stopped at the declared KL limit. The frozen earlier v4 weights repeated a positive direction on the larger panel (+5.21 points) but failed the same criteria. After the additional updates, v4 won140 against121 in192 cases: +9.90 percentage points,38 paired gains/19 losses,p=0.00817. This met the statistical threshold but missed the frozen10-point minimum by one win. The failed verdict is retained and its independent final stayed unopened. No weights were promoted.

## Curriculum coverage and verification

The fixed `pressure` simulator opponent uses its own request and public battle view. An explicitly declared synthetic team pool exercises setup, redirection and HP-dependent attacks; those sets are not reconstructed hidden ladder opponents. Actual executions verify exposure rather than inferring it from a team list. The Volcarona fixture exposed a selection gap: its pressure policy instead brought Milotic/Rillaboom/Garchomp/Incineroar in nine audited, fully revealed games, so Quiver Dance was not exercised there. The source/pool remained fixed during the declared experiment.

The reviewed public source passed259 tests, including native-mechanics, continuation, pressure, provenance and archival checks. Replay across all100 reviewed games/900 decisions reproduced the incumbent action arrays exactly, preserved state arrays within tolerance and matched collecting likelihoods below0.0001. This establishes compatibility for those recorded cases, not competitive superiority of the new profiles.

A candidate requires a complete clean development gate and an independent final gate before staging. Source generation, parent revision and checkpoint hash must match; promotion occurs before matchmaking between recorded battles. The currently selected live model remains strategic-v1. Any prospective confirmation must declare its criteria before inspecting new cases; the completed10-point-gate verdicts stay unchanged.

The trained v4 stratum results were: self55 cases26→35 wins; pressure44 cases21→32; tactical49 cases36→35; heuristic22 cases17→16; random22 cases21→22. The aggregate improvement does not establish superiority in every opponent group, and the curriculum coverage gap remains explicit.

## Fresh confirmation after the minimum-gain decision

The operator prospectively selected a5-percentage-point minimum while retaining the one-sidedp<=0.05 threshold. The completed10-point verdicts were left unchanged. A new200-case development panel used the immutable trained weights, new seeds, the same source/team pool and matching frozen scouting/research content. It finished139–136 (+1.5 points,p=.38988), failing both the practical and statistical criteria. Its independent200-case final remained unopened, and the live weights were retained.

Fresh strata: self55 cases27→31; tactical55 cases45→38; pressure46 cases26→29; heuristic22 cases16→19; random22 cases22→22. The earlier9.9-point result did not replicate. The evidence supports investigating the tactical-opponent failures, not deploying the candidate or pooling panels after the fact to manufacture a pass. Selected replay diagnoses stay separate from PPO/expert labels.

## V5 source and paused experiment

The opt-in `strategic-v5` profile adds defense/HP-aware preview coverage and
bounded hypotheses for undisclosed opposing Mega stones. Format eligibility is
explicit uncertainty, never a claim that an opponent holds a particular item.
An explicit empty or removed item excludes those hypotheses. One prospective
own Mega is considered at a time. Legacy collecting profiles remain unchanged.

The prospective v5 run froze a 756-game corpus (7,260 observed move executions),
13-team pool, four 128-game fresh training rounds and separate 200-case
development/final panels, with a 5-point minimum and paired p≤0.05. At shutdown,
two 128-game updates were complete; the third collection round had 51/128 games.
The experiment is paused and retained locally. No completed development/final
gate or v5 competitive improvement is claimed, and no v5 weights were promoted.
Training plans now bind their runner hash and prospective minimum gain, so a
resume cannot silently change the declared criteria.
