# VLM regression fixture

Adversarial keyframe fixture used to detect prompt regressions, model-version
drift, and pre-VLM gate misfires before they reach production. Run on every
deploy via `pytest tests/regression/test_vlm_regression.py`.

## Layout

```
tests/regression/vlm_fixture/
├── empty_cab/         # cabin has no person → must FP for writing/eating/etc.
├── idle_person/       # LP seated, hands in lap, no object → must FP for writing
├── writing_tp/        # LP demonstrably writing in log book → must TP for writing
├── writing_fp/        # writing-shaped FPs (brake handle, radio, watch glance, …)
└── no_object_writing/ # person + writing pose but no rendered book bbox → must
                      #   trigger the pre-VLM no-object gate (PRE_GATE_DROP_NO_OBJECT)
```

Each fixture directory contains one or more `.jpg` keyframes WITH the
Pipeline-1 overlay (green person bbox, orange/yellow object bbox, magenta
bag bbox). For `empty_cab` and `no_object_writing` fixtures, the keyframe
must visibly demonstrate "no person" or "no object" so the deterministic
gate can be tested without a vLLM endpoint.

## Adding a new fixture

1. Drop the `.jpg` into the relevant directory (rename for readability:
   `tv22_10_run20260509_act3_supervisor_visit.jpg`).
2. Add an entry to `expectations.json` in the same directory:

```json
{
  "filename": "tv22_10_run20260509_act3_supervisor_visit.jpg",
  "object_type": "writing",
  "expected_gate": "PRE_GATE_DROP_NO_SUBJECT",
  "expected_verdict": null,
  "comment": "supervisor visit, 3 standing crew, no LP/ALP writing"
}
```

`expected_gate` is one of `PRE_GATE_DROP_NO_SUBJECT`,
`PRE_GATE_DROP_NO_OBJECT`, or `null` (gate doesn't fire).
`expected_verdict` is the VLM verdict expected when the gate doesn't fire
(`TRUE_POSITIVE` / `FALSE_POSITIVE` / `UNCERTAIN`), or `null` when only
the gate is exercised.

## CI policy

The regression test SKIPS gracefully when the fixture directory is empty
or when no `expectations.json` exists, so the suite passes in a fresh
clone. Once fixtures are added, the test runs and the deploy gate enforces
≥ 95% precision on `empty_cab` + `idle_person` + `writing_fp` + `no_object_writing`,
and ≥ 85% recall on `writing_tp`.

## Bootstrapping

The recommended starter set:

- 5–10 frames each of `empty_cab` (mined from `run_*` directories where
  Pipeline-1 fired writing on no-person frames — `vlm_disagreements.jsonl`
  is the right place to look)
- 5–10 frames each of `writing_tp` (clear, unambiguous writing — first 10
  hand-labelled positives)
- 10–20 frames of `writing_fp` (one per known confounder: brake handle,
  radio handset, watch glance, idle lap, static book, supervisor visit)
