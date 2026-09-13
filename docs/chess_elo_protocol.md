# Chess Elo experimental protocol

## Goal

Measure whether a frozen MaleCNS-based recurrent system becomes a stronger chess player when we change a controlled variable such as recurrent depth, dynamics, trainable readout, or physiology.

The primary metric is estimated Elo against calibrated Stockfish opponents.

## Opponent

Use a fixed Stockfish release and record its version. Configure:

- `UCI_LimitStrength = true`
- `UCI_Elo = <target integer>`
- fixed Threads and Hash
- no pondering

The harness checks the installed engine's actual `UCI_Elo` min/max at runtime. The target is a calibrated nominal rating, not a mathematical guarantee of realized Elo under arbitrary time controls. Final experiments should use one fixed time control and must not compare runs made with different Stockfish versions/settings.

Current Stockfish builds commonly expose a lower UCI_Elo near 1320. If the fly is weaker than the engine's minimum, report the result as **below the measurable Stockfish range** rather than fabricating a numeric Elo.

## Fly move selection

The fly does not receive one dedicated output neuron per chess move.

For each legal candidate move:

1. Encode the current board and candidate move into a fixed chess feature vector.
2. Project that vector into a frozen set of sensory neurons.
3. Run the same connectome for `D` recurrent passes.
4. Read one scalar score from a frozen readout population.
5. Play the legal move with the highest score.

The candidate interface makes the number of legal moves irrelevant and keeps the connectome architecture fixed.

## What must be frozen during a depth experiment

To attribute Elo change to recurrent depth, freeze all of the following:

- MaleCNS graph and connection filters
- neuron dynamics and gains
- chess feature definition
- sensory neuron set
- feature-to-sensory projection seed and fanout
- readout neurons and readout weights
- training set/checkpoint
- Stockfish version/settings
- opening suite and game-count policy

Only the tested depth parameter changes.

## Opponent ladder

Do not test against one Elo only. Use a bracket around the fly's current strength. A typical first ladder is:

`1320, 1400, 1500, 1600, 1700`

After locating the approximate rating, narrow the ladder around the 30-70% score region. This provides substantially more information per game than playing opponents that always win or always lose.

## Game counts

Recommended tiers:

- smoke: 20-40 games total
- development comparison: 100-200 games total
- serious estimate: 400+ games total
- publication-quality comparison: use sequential power analysis and paired openings

Near a 50% score, 400 independent games give a classical-Elo local 95% uncertainty of roughly +/-34 Elo before accounting for correlation and opening effects.

## Color and openings

At minimum, alternate colors exactly. For serious experiments use paired openings: play each opening twice, once with the fly as White and once as Black. Reuse exactly the same opening suite for every model/depth comparison.

## Elo estimator

Given known opponent ratings R_i and fly scores s_i in {0, 0.5, 1}, estimate R by solving:

`sum_i [s_i - E(R, R_i)] = 0`

where

`E(R, R_i) = 1 / (1 + 10^((R_i - R)/400))`.

The repository reports a local Fisher-information interval as a quick uncertainty estimate. For final scientific claims, add paired-opening bootstrap intervals.

## Critical interpretation

Chess Elo is a benchmark for the **complete frozen system**: connectome + dynamics + chess encoder + readout. It is not automatically the biological fly's Elo.

The most informative experiment is differential:

`same checkpoint + same adapter + same games, varying only recurrent depth`.

If Elo rises reproducibly with depth and beats shuffled-topology / shuffled-sign controls, that is evidence that the connectome is exploiting additional recurrent computation on this task.
