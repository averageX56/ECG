# Qwen pseudo preparation: cluster performance

Primary runtime: Linux A100 / JupyterLab, see [CLUSTER](CLUSTER.md).
Existing raw → CPU workers → batched CUDA U-Net → persistent pseudo cache.
Optional `QWEN_SCRATCH` adds a node-local cache and final validated sync to artifacts.
The cluster launcher never deletes, moves, copies or relinks existing raw directories.
Legacy Colab helpers remain for compatibility; the notebook does not import them.

Defaults: batch64, spawn workers8. Only the parent owns CUDA/the teacher.
Fresh signals are decoded once; only the selected lead is filtered, preserving time alignment.
Separate progress bars: protection registry, preprocessing, inference/write, optional final sync.
Preprocessing progresses before the first CUDA batch is ready.
Qwen E sources: CPSC_EXTRA, PTBXL, CPSC, CHAPMAN. Ningbo remains available to other branches.
LUDB/QTDB controls and protected record cohorts remain mandatory for patient/record/exact-signal exclusion.

Resume verifies NPZ hashes and metadata before raw decoding. Teacher hash, cohort identity,
preprocessing version, tau and cached signal identity remain bound to the cache.
Unchanged source size/mtime enables fast reuse; `VERIFY_QWEN_SOURCES=True` rehashes raw contents,
including changes preserving timestamps. Legacy shards are fully hashed once to migrate metadata.
Protected identity shards use the same fast path. Stale caches fail instead of changing cohorts silently.

Only new outputs use scratch. Existing partial persistent caches resume in place.
Completed caches are validated before final sync; provenance is published last via a sibling `.syncing` directory.
Interrupted copies cannot appear as a complete destination. Different existing outputs are preserved and rejected.
No periodic sync or automatic cleanup; choose durable output if the job may lose its local SSD.

Previously idle CUDA could be waiting for Drive I/O, preprocessing and batch256. Shared cluster storage
can also be the bottleneck. Actual A100 throughput is unmeasured here; use `preparation_run.json`
(records/sec and PyTorch peak allocated/reserved memory) and stage progress.
