# Codex task: E1.2

Execute in this order without running test:

1. Read `README.md`, `AGENTS.md`, and `docs/E1_2_IMPLEMENTATION_SPEC.md`.
2. Run `ruff check src tests scripts` and `pytest -q`; fix all E1.2 failures without changing the research contract.
3. Run toy-cache train/evaluate smoke tests for `train_e12` and `evaluate_e12`.
4. Run `scripts/e12_oracle_ap_audit.py` on validation and connect both O1/O2 outputs to the existing project COCO evaluator.
5. Reproduce T0 from exactly the same cache and evaluator.
6. Run T1–T6 with seeds 41/47/53. Use config overrides:
   - T1: `data.include_nms=true data.include_low_conf=false`;
   - T2: `data.include_nms=false data.include_low_conf=true`;
   - T3+: both true;
   - RGB-only: `data.include_rgb=true data.include_pol=false`;
   - POL-only: `data.include_rgb=false data.include_pol=true`.
7. Save validation operating curves and per-image restore logs.
8. Verify every image preserves safe boxes/classes and `restore_count <= 1`.
9. Fill AP50/AP75/AP50:95 with the existing COCO evaluator; do not approximate AP in `search_e12.py`.
10. Only create `thresholds.locked.yaml` when all four gates pass across the required seeds.

Required gates:

```text
AP50:95 >= 0.4547
Recall >= 0.8596
FP/image <= 0.248
Destruction <= 0.0251
```

Do not run test or start 14/14 detector OOF unless debug validation passes the complete gate set.
