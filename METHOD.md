# How the model works

The chain runs from the betting market down to an individual player's carry:

```
market total + spread
      └─► implied team points
            └─► expected offensive touchdowns
      └─► projected plays  ──► pass / run split      ← scheme fingerprint
                                    └─► player shares ← depth chart + usage history
                                          └─► yards  ← bootstrapped per-touch outcomes
                                          └─► TDs    ← red-zone and goal-line shares
                                                └─► × P(active)
```

Everything is simulated, not point-estimated. The useful questions are
distributional — the chance a back clears 70 yards, the chance a receiver scores
— and a single number cannot answer them.

## Data

All of it is public and fetched from [nflverse](https://github.com/nflverse/nflverse-data):

| Source | What it provides |
| --- | --- |
| Play-by-play (2022–2025) | Every snap: down, distance, field position, EPA, air yards, run gap, pass depth |
| FTN charting (2022–2025) | Pre-snap motion, play-action, RPO, screens, blitzers, pass rushers, box count |
| Snap counts (2023–2025) | Who was actually on the field, and how often |
| Depth charts (2026) | Current roles, refreshed through the preseason |
| Rosters, players | Identity and position, plus the id crosswalk between feeds |
| Schedules + market lines | Opponent, rest, closing total and spread |

Charting is what makes scheme measurable rather than inferred. Without it you
can see that a team ran the ball; with it you can see that they ran it from
shotgun, out of jet motion, to the right edge.

## Which games count as evidence

**Preseason is excluded unconditionally**, and that is not configurable.
Starters play a series and sit, most snaps go to players who will not make the
roster, play-calling is deliberately vanilla, and nobody is trying to win. It
measures how individual players look in isolation, not how a team will operate
in September — so weighting it at all would import noise dressed as signal.

nflverse does not currently publish preseason play-by-play, so today this filter
removes nothing. It is enforced anyway: if a future feed change starts shipping
preseason rows, they must not quietly enter the model. A test asserts preseason
stays out even when postseason is explicitly requested.

**Postseason is excluded by default**, for a different reason. It is not
unrepresentative — it is the most serious football played — but it is
*asymmetric*. Only fourteen teams have any, so including it hands extra sample
to teams that were already good, gathered against stronger opponents in
higher-leverage scripts, and gives everyone else nothing. That is a bias in the
comparison between teams rather than a bias in any one team's numbers. Set
`INCLUDE_POSTSEASON = True` in `config.py` to use it.

The one place preseason legitimately shows up is the **depth chart**, which is
refreshed through August. That is a statement of intended role, not a
performance measurement, and intended role is exactly what the usage model
needs. It is also the model's most fragile input: chart positions move through
cutdowns and into September, and a stale chart is the likeliest cause of a badly
wrong player projection.

## Scheme fingerprints

Each team-season reduces to a tendency vector: early-down pass rate, pass rate
over expected, motion, play-action, RPO and screen rates, shotgun, tempo, depth
of target, run direction and gap, red-zone and goal-to-go tilt, fourth-down
aggression. Defensively: blitz rate, pass rushers, box counts, and what the unit
surrenders.

These separate cleanly along lines a coach would recognise. The 2025 Rams motion
on 58% of snaps and run under-center play-action deep shots at nearly three
times the league rate; Miami's jet-motion outside zone runs at four and a half
times league rate; Minnesota blitzes on 46% of dropbacks.

## Carrying scheme across a coaching change

Ten head-coaching jobs and roughly twenty coordinator jobs changed hands before
the 2026 season, which breaks the usual assumption that last year predicts this
one. The model blends three sources:

```
projection = w_coach  × play caller's fingerprint at their previous stop
           + w_team   × this team's own fingerprint last season
           + w_league × league mean
```

`w_coach` rises with how much NFL evidence the coach has, whether they call
plays themselves, and how well corroborated the hire is. A first-time coordinator
with no NFL sample falls back to the system they came from, at reduced weight; if
even that is missing, the projection stays on team continuity and regresses
harder.

Who coaches where lives in **`data/coaching_2026.yaml`**, which is meant to be
edited. Coaching news moves faster than any data feed, and several aggregators
disagreed with each other while this was built, so every team carries a
`confidence` flag that controls how far the projection is allowed to travel from
last season's baseline. Correct a line in that file and the projections change
immediately; nothing else hardcodes a coach.

## Personnel constrains scheme

A fingerprint describes what a coach did *with different players*. Some of it
travels and some does not, so personnel-bound traits are pulled toward the
players actually on the roster:

| Trait | Player-driven |
| --- | --- |
| Designed QB run rate | 85% |
| Scramble rate | 80% |
| Sack rate allowed | 45% |
| Depth of target, deep rate | 40% |
| YAC share | 30% |

The clearest case: Todd Monken's Baltimore offense ran designed quarterback runs
at 4.2% of snaps, because Lamar Jackson was taking the snaps. Carried to
Cleveland unmodified, the model would hand that rate to a pocket passer. The
personnel layer drops it to 1.4%. In the other direction, Baltimore keeps a high
designed-run rate under a brand-new coordinator, because Jackson is still there.

## Usage

Shares come from play-by-play, not box scores, so red-zone targets and goal-line
carries are visible — that is where touchdowns actually live.

Three corrections matter:

**Availability, not role.** Shares are computed per game *played*. A receiver who
holds a 29% target share for four games and then tears something held a
starter's role; season totals alone would score him as a reserve.

**Shares do not travel.** A share is team-relative — it describes how one offense
divided its work, against that particular set of competitors for touches. When a
player changes teams, past share is weak evidence and the role prior takes over.

**The depth chart states intent.** A back who carried on 55% of his old team's
runs and is now listed second is being told something. Where a player's history
implies a different role than the one he now holds, the projection regresses
toward the prior for the assigned role.

Role priors are measured, not guessed: ranking every team-season's players within
position across 2022–2025 gives WR1 = 23.1% of targets, RB1 = 53.9% of carries,
RB1 = 55.3% of goal-line carries, and QB1 = 11.6% of goal-line carries — which is
why mobile quarterbacks eat into their lead back's rushing touchdowns.

**Air-yards share** carries the majority of the target projection. Measured
year-over-year, it is the most stable usage signal available:

| metric | stability |
| --- | --- |
| Share of intended air yards | **0.727** |
| Avg separation | 0.547 |
| Target share | ~0.55 |
| Catch % | 0.432 |
| YAC above expected | 0.256 |

Air yards are targets multiplied by depth, so dividing a projected air-yards
share by a receiver's own depth of target recovers an independent target count —
anchored to a stronger role signal than target share alone. It also encodes
*what kind* of receiver someone is: Alec Pierce took 42% of Indianapolis's air
yards on 18% of its targets, which is a different asset from a possession
receiver with the same target share.

**Draft capital** is the fallback for a player with no NFL sample, and it fades
to nothing the moment real usage exists. **Separation** nudges catch rate, being
a more stable skill signal than catch rate itself.

**Quarterback continuity** is tracked but deliberately not used to move the mean
much. When a receiver has barely played with his projected quarterback, that
pairing is *unmeasured*, not bad — so the model widens its regression toward the
role prior and reports the shared-target count so the uncertainty is visible.

## Yards

Per-touch gains are bootstrapped from real play-by-play outcomes rather than
drawn from a fitted normal. Rushing and receiving gains are heavily skewed —
most carries gain three yards and a few go eighty — and that tail is exactly
what decides a yardage line. Separate pools are kept for short, intermediate and
deep receiving work, then scaled toward each player's own efficiency and the
opponent's defense.

## Touchdowns

Team touchdowns come from implied points via a fit on 2022–2025 team-games
(`TD = -0.463 + 0.1255 × points`, r = 0.886), split between pass and run using
the team's own red-zone and goal-to-go tendencies. Player scoring share weights
red-zone targets and goal-line carries far more heavily than overall volume,
because that is where scoring is decided.

## Availability

Backtesting surfaced the single largest error in the model. Projecting every
listed player as if he suits up every week overstated volume by 25–40% and
anytime-touchdown probability by five points — because **46% of projected
player-games ended with no touches**. Conditional on playing, the same
projections were close to unbiased.

So availability is modelled explicitly, from snap-count history regressed toward
a role prior, and applied as a per-simulation Bernoulli draw. That puts a real
point mass at zero instead of quietly shrinking every number. Both views are
reported: **if active** (what he does when he plays) and **expected** (what he is
worth once availability is priced in). A bet settles on the expected view.

## Validation

`scripts/backtest.py` holds out 2025 entirely. Fingerprints, usage, efficiency
and availability are built from 2022–2024 only; the 2025 schedule, its market
lines and its depth charts are supplied; projections are compared to what
actually happened. Every team is treated as coaching-continuity in the backtest,
so it measures the projection chain rather than the coaching research.

## Game predictions

The player projections *consume* the market line. The game model does not — it
forms an independent view so the two can be compared.

Team strength is estimated by ridge regression on play-level EPA
(`epa ~ offense[team] + defense[opponent] + home`), which separates a team's own
quality from the schedule it faced. Ratings are then carried forward with
side-specific shrinkage, because the two sides of the ball behave completely
differently:

| | year-over-year correlation |
| --- | --- |
| Offensive EPA | 0.44 |
| Defensive EPA | **0.12** |

A defence is close to a coin flip from one season to the next, so defensive
ratings are pulled hard toward average rather than projected forward. Teams with
a new play caller are shrunk further still. During the season, ratings are
refreshed from games already played and blended against the preseason prior by
how much has been seen.

Margin and total are simulated on separate calibrated axes rather than by
drawing each team's score independently — the two scores in a game are
correlated through pace and script, and that shared component *cancels* in the
margin. Simulating team scores separately understated margin variance by about
two points of standard deviation and made win probabilities visibly
over-confident.

### How good is it, honestly

Held out over 544 games (2024–2025), predicting each week from data available
before it:

| | model | market closing line |
| --- | --- | --- |
| Margin MAE | 10.26 | **9.67** |
| Margin correlation | 0.42 | **0.50** |
| Total MAE | 10.19 | **10.06** |
| Straight-up winners | 64.7% | **68.4%** |

**The market is better on every measure.** Betting the model's disagreements
lost money: 48.7% correct on edges of two points or more, against a 52.4%
break-even at standard vig. A large edge is best read as a flag that the model
is missing something the market knows — usually injury or personnel news — not
as a signal to act on.

The model does clear the naive baselines: 10.26 margin MAE against 11.17 for
always picking a tie, and 64.7% winners against roughly 53% for always taking
the home team. So it knows something. It just knows less than the closing line.

One correction was tested and rejected. The model's margins are compressed
toward zero — regressing actual margin on predicted gives a slope of 1.45, so
scaling predictions up looks like an easy gain. Fitting that scale on 2024 and
applying it to 2025 improved MAE by 0.04 points and made straight-up accuracy
*worse* (63.6% → 62.9%), and the slope itself moved from 1.54 to 1.34 between
seasons. The market shows the same apparent compression in this sample, which
suggests it is a property of these two seasons rather than a fixable model
defect. It was left out.

## Coverage

Front and pressure are only half a defensive call. nflverse participation
charting supplies the other half — man or zone, and the shell — labelled on
roughly every charted pass play (about half of all snaps). It also carries the
route each receiver ran and whether the quarterback was pressured.

This separates defences that the front-based fingerprint treats as similar.
Minnesota blitzes on 46% of dropbacks but plays man on only 18% of snaps and
sits two-high 59% of the time: pressure with coverage behind it. Denver plays
man on 45%, single-high on 61%, and Cover 0 on 6.5%. Those are opposite
philosophies that a blitz rate alone would not distinguish.

The same data measures how an offence fares against each structure, which turns
a matchup into a specific read — *Kansas City are better against zone (+0.150
EPA against −0.063 versus man) and Denver play man on 45% of snaps against a 31%
league rate* — rather than a generic strength comparison. Route distributions and
offensive personnel groupings (11, 12, 21) come from the same feed.

## Blocking versus the back

A back's yards per carry blends two things that behave nothing alike:

| | year-over-year correlation |
| --- | --- |
| Yards **before** contact per carry (blocking) | **0.433** |
| Yards **after** contact per carry (the back) | 0.106 |

Blocking persists; broken tackles largely do not. Regressing raw yards per carry
toward a league mean treats them as one quantity and discards the half that is
actually predictable. They are now projected separately — the line's
contribution from the team, the back's from the player with heavy regression —
and reported separately, so a projection can say whether a back is producing
behind good blocking or creating on his own. Only one of those travels if he
changes teams.

The spread is not small. In 2025 Chicago generated 3.31 yards before contact per
carry and Las Vegas 1.63.

## Kicking

Field goal probability is a logistic in distance fitted on 4,325 regular-season
attempts: `logit(make) = 5.915 − 0.0985 × distance`, putting a 25-yarder at 97%,
a 45-yarder at 82% and a 58-yarder at 55%. A kicker's own record moves this only
modestly — a season is about thirty attempts, which is not much signal.

Wind is **not** applied to accuracy. Controlling for distance the coefficient is
−0.017 per mph with a bootstrap *t* of −1.0 and a 95% interval spanning zero, on
only ~200 attempts in 15+ mph conditions. Coaches also attempt shorter kicks in
wind, absorbing part of the effect at the decision level. Wind still reduces a
kicker's output, through fewer trips into range rather than worse kicking.

## Situational context: travel, weather, and familiarity

Four factors were tested — travel distance and time zones, body clock at
kickoff, weather, and divisional familiarity. Each was measured over **7,276
games with a closing line (1999–2025)**, and checked for stability across three
eras. Some are real. Most of the famous ones are not, any more.

The test is deliberately strict. Performance is measured against the *closing
spread*, because the market already prices team quality — a West Coast team
losing in the East proves nothing if it was an underdog. The question is whether
it loses by more than the line expected.

### What is applied

| Effect | Size | Evidence |
| --- | --- | --- |
| Wind | **−0.19 pts of total per mph** above 8 | t = −7.5 raw; holds in all three eras |
| Indoor venue | **+1.0 pts** of total | t = +9.4 raw; holds in all three eras |
| Divisional game | **−0.91 pts** of total | t = −4.4 raw; consistent sign throughout |

Wind is the weather variable that matters, and it is not subtle. Measured over
7,901 charted plays in 15+ mph conditions against a calm outdoor baseline,
offenses throw less (pass rate −2.4 points), throw shorter (deep rate −14%
relative), and complete fewer (−2.2 points). Those changes flow into the player
projections, which is why a windy game moves a quarterback's yardage down and
his running back's up.

Cold, notably, does almost nothing on its own. Controlling for the market total,
each degree is worth −0.009 points — a 40-degree swing moves a game by about a
third of a point. Freezing games do not reliably go under.

### What is deliberately not applied

Travel distance, time zones crossed, and visiting-team body clock at kickoff are
computed and displayed, but carry **no weight** in any projection. They were
tested and failed:

| Split | 1999–2009 | 2010–2017 | 2018–2025 |
| --- | --- | --- | --- |
| East team travelling west, 2+ zones | **−2.79** (t = −3.6) | −0.11 | +0.24 |
| Dome team playing outdoors | **−1.49** (t = −2.6) | −0.12 | +0.56 |
| West team travelling east, 2+ zones | +0.22 | +0.11 | +1.69 |

The east-to-west effect was real and large a generation ago. It is gone. That is
what an efficient market does to a publicised edge, and a model that still paid
for it would be betting on a pattern that stopped existing. The famous "West
Coast team in a 1pm Eastern kickoff" body-clock effect does not appear at all
(t = −0.58 across the full sample), and the raw version of it has the *opposite*
sign to the folklore, because West Coast teams have simply been good.

Divisional games are also less special than they sound. They are lower-scoring,
which is applied — but against the spread the effect washes out in the modern
era (t = 1.05 since 2010), and divisional games are *not* closer than the line
expects: average absolute margin 11.36 against 11.65 for non-divisional.

### A note on "rivalry"

There is no rivalry variable. Divisional status is the only objective, complete
definition available; anything beyond it (Packers–Bears being special in a way
Packers–Vikings is not) would be a hand-built list encoding my own assumptions,
and it would not be measurable. Divisional games are what the data can speak to.

### Weather availability

Weather is only published once a game has been played, so a season-ahead
projection has none, and the model correctly applies no adjustment rather than
inventing one. Roof and divisional status *are* known in advance and always
apply. The app exposes a wind slider so a forecast can be entered by hand during
game week.

## What this does not do

- **The coaching registry is research, not a feed.** It was assembled from
  reporting that in several cases contradicted itself. Check it before trusting
  a team whose staff turned over, and edit the YAML where it is wrong.
- **Depth charts are preseason.** Roles change through cutdowns and into
  September, and a stale chart is the most likely source of a badly wrong player
  projection.
- **Offensive line quality is a rushing and pressure proxy, not a unit model.**
  Yards before contact and pressure allowed stand in for blocking; there is no
  per-lineman projection, so a line that loses a starter is not modelled.
- **Coverage is charted on pass plays only**, so about half of snaps carry a
  label, and the profiles describe last season's staff. Where a defensive
  coordinator changed, the coverage profile is the *old* staff's and should be
  read alongside the scheme-transfer weighting.
- **Rookies get a role prior scaled by draft capital**, and nothing else — no
  combine or college production translation is attempted.
- **Market lines are only posted for part of the season.** Later weeks fall back
  to a generic game environment, which flattens strength of schedule.
- **Defensive personnel is not projected.** Defensive scheme is carried across
  coaching changes, but without a unit-quality projection to go with it.
- **No travel or body-clock adjustment**, by choice — see above. The effects
  were real historically and are not any more.
- **No true rivalry variable**, only divisional status.
- **Weather must be entered by hand** for future games; it is not forecast.
- **The game model does not beat the market**, and is not built to. It has no
  access to injury reports, weather, or news, and it has no mechanism for the
  in-week information that moves a line.
