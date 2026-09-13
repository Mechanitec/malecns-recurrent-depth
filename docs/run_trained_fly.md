# Run a trained MaleCNS fly in the chess benchmark

`run_chess_benchmark.py` can now reconstruct a `FlyCandidateMoveAgent` directly from a trained `readout_checkpoint.npz`.

The checkpoint supplies the trained readout weights/indices plus the frozen chess-interface metadata recorded during activation extraction: recurrent depth, dynamics type, projector seed/fanout/amplitude, and clamp mode. The MaleCNS graph itself is reloaded from the same flat-file dataset used during training.

## Required inputs

- `readout_checkpoint.npz`
- MaleCNS annotations Feather file
- MaleCNS neurotransmitters Feather file
- MaleCNS connectome weights Feather file
- the `.npy` sensory-index vector used during activation extraction
- Stockfish executable
- Alfil executable when testing below the Stockfish Elo floor

## Example

```bash
python scripts/run_chess_benchmark.py \
  --checkpoint results/train_depth16/readout_checkpoint.npz \
  --annotations data/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --neurotransmitters data/body-neurotransmitters-male-cns-v1.0.feather \
  --connectome-weights data/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --sensory-indices data/chess_sensory_indices.npy \
  --stockfish /path/to/stockfish \
  --alfil /path/to/alfil \
  --elos 0,200,400,600,800,1000,1200,1320,1400 \
  --games-per-elo 20 \
  --analysis-depth 18 \
  --output results/fly_depth16_elo
```

Without `--checkpoint`, the CLI deliberately falls back to `RandomLegalAgent` as a tournament-plumbing smoke test.

## Recurrent-depth experiment

The strongest causal test keeps the trained checkpoint fixed and changes only inference depth:

```bash
--fly-depth 1
--fly-depth 2
--fly-depth 4
--fly-depth 8
--fly-depth 16
--fly-depth 32
--fly-depth 64
```

All other graph, projector, readout, opponent, opening, and analysis settings should remain identical across these runs.

## Live analysis

Position evaluation is enabled by default. A separate full-strength Stockfish process evaluates the initial board and every ply while the requested Alfil/Stockfish opponent continues to play at its configured strength.

Launch the dashboard with:

```bash
streamlit run dashboard/app.py
```

Then open the **Live Analysis** page to see the fly icon, board, neural candidate scores, current Stockfish evaluation, and evaluation history.
