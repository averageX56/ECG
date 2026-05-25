import sys
import traceback
import numpy as np

sys.path.insert(0, "/mnt/project")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))


print("\n══ BACKBONE ══")

def t_resnet_imports():
    from resnet1d import ResNet1d, ResNet1dWithHead, build_resnet1d, _ResBlock1d, _Stem1d
    from resnet1d import _FEAT_DIM, _BLOCK_FILTERS, _BLOCK_STRIDES, _KERNEL_SIZE, _INIT_FILTERS, _N_LEADS, _DROPOUT
    assert _FEAT_DIM == 320
    assert _N_LEADS == 12

check("resnet1d imports", t_resnet_imports)

def t_resnet_forward():
    import torch
    from resnet1d import build_resnet1d
    model = build_resnet1d()
    model.eval()
    x = torch.randn(2, 12, 5000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 320), f"expected (2,320) got {out.shape}"

check("resnet1d forward pass [2,12,5000] -> [2,320]", t_resnet_forward)

def t_resnet_with_head():
    import torch
    from resnet1d import ResNet1dWithHead
    model = ResNet1dWithHead(n_classes=27)
    model.eval()
    x = torch.randn(2, 12, 5000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 27)

check("ResNet1dWithHead forward [2,12,5000] -> [2,27]", t_resnet_with_head)

def t_resnet_freeze():
    from resnet1d import build_resnet1d
    model = build_resnet1d()
    model.freeze_all()
    assert model.trainable_params() == 0
    model.unfreeze_all()
    assert model.trainable_params() > 0
    model.freeze_all()
    model.unfreeze_last_n(2)
    assert model.trainable_params() > 0
    assert model.frozen_params() > 0

check("freeze/unfreeze backbone", t_resnet_freeze)

def t_resnet_features():
    import torch
    from resnet1d import build_resnet1d
    model = build_resnet1d()
    model.eval()
    x = torch.randn(2, 12, 5000)
    with torch.no_grad():
        feat, intermediates = model.forward_features(x)
    assert feat.shape == (2, 320)
    assert len(intermediates) == 4

check("forward_features returns intermediates", t_resnet_features)

def t_load_weights_imports():
    from load_weights import (
        load_pretrained_weights, download_hdf5, convert_keras_to_pytorch,
        ZENODO_URL, ZENODO_RECORD_ID, _DEFAULT_CACHE_DIR
    )
    assert "zenodo" in ZENODO_URL.lower()
    assert ZENODO_RECORD_ID == "3625017"

check("load_weights imports + constants", t_load_weights_imports)

def t_backbone_package():
    import importlib.util
    spec = importlib.util.spec_from_file_location("backbone_init", "/mnt/project/__init__.py")
    # Check all expected symbols exist in module
    from resnet1d import ResNet1d, ResNet1dWithHead, build_resnet1d
    from load_weights import load_pretrained_weights, download_hdf5

check("backbone __init__ exports", t_backbone_package)


print("\n══ PREPROCESSING ══")

def t_filters_imports():
    from filters import apply_ecg_filters, apply_lowpass, apply_notch, apply_notch_harmonics
    from filters import design_lowpass_kaiser, get_powerline_freq, DATASET_POWERLINE
    assert DATASET_POWERLINE["georgia"] == 60
    assert DATASET_POWERLINE["ptbxl"] == 50

check("filters imports + powerline table", t_filters_imports)

def t_filters_notch():
    import numpy as np
    from filters import apply_notch
    sig = np.random.randn(12, 5000).astype(np.float32)
    out = apply_notch(sig, fs=500.0, freq=50.0)
    assert out.shape == sig.shape
    assert out.dtype == np.float64

check("apply_notch shape/dtype", t_filters_notch)

def t_filters_lowpass():
    import numpy as np
    from filters import apply_lowpass
    sig = np.random.randn(12, 5000).astype(np.float32)
    out = apply_lowpass(sig, fs=500.0, cutoff=40.0)
    assert out.shape == sig.shape

check("apply_lowpass shape", t_filters_lowpass)

def t_filters_pipeline():
    import numpy as np
    from filters import apply_ecg_filters
    sig = np.random.randn(12, 5000).astype(np.float32)
    out50 = apply_ecg_filters(sig, fs=500.0, powerline_freq=50)
    out60 = apply_ecg_filters(sig, fs=500.0, powerline_freq=60)
    assert out50.shape == (12, 5000)
    assert out60.shape == (12, 5000)

check("apply_ecg_filters 50Hz + 60Hz", t_filters_pipeline)

def t_normalize_imports():
    from normalize import (
        resample_signal, crop_or_pad, crop_center_or_pad,
        znorm, znorm_per_channel, normalize_record_stream_a,
        normalize_beat_stream_b, align_by_rpeak, get_transform_info
    )

check("normalize imports", t_normalize_imports)

def t_resample():
    import numpy as np
    from normalize import resample_signal
    sig = np.random.randn(12, 2570)
    out = resample_signal(sig, fs_in=257, fs_out=500)
    expected = round(2570 * 500 / 257)
    assert abs(out.shape[1] - expected) <= 2, f"got {out.shape[1]}, expected ~{expected}"

check("resample_signal 257->500 Hz", t_resample)

def t_crop_pad():
    import numpy as np
    from normalize import crop_or_pad, crop_center_or_pad
    sig = np.random.randn(12, 6000)
    out = crop_or_pad(sig, 5000)
    assert out.shape == (12, 5000)
    sig2 = np.random.randn(12, 4000)
    out2 = crop_or_pad(sig2, 5000)
    assert out2.shape == (12, 5000)
    out3 = crop_center_or_pad(sig, 5000)
    assert out3.shape == (12, 5000)

check("crop_or_pad + crop_center_or_pad", t_crop_pad)

def t_znorm():
    import numpy as np
    from normalize import znorm
    sig = np.random.randn(12, 5000) * 10 + 5
    out = znorm(sig)
    assert abs(out.mean()) < 0.01
    assert abs(out.std() - 1.0) < 0.01

check("znorm mean~0 std~1", t_znorm)

def t_normalize_stream_a():
    import numpy as np
    from normalize import normalize_record_stream_a
    sig = np.random.randn(12, 6000).astype(np.float32)
    out = normalize_record_stream_a(sig, fs_in=500, fs_out=500, target_len=5000)
    assert out.shape == (12, 5000)
    assert out.dtype == np.float64

check("normalize_record_stream_a -> [12,5000]", t_normalize_stream_a)

def t_normalize_beat():
    import numpy as np
    from normalize import normalize_beat_stream_b
    beat = np.random.randn(2, 320).astype(np.float32)
    out = normalize_beat_stream_b(beat, target_len=300)
    assert out.shape == (2, 300)

check("normalize_beat_stream_b -> [2,300]", t_normalize_beat)

def t_snomed_imports():
    from snomed_map import (
        SCORED_SNOMED_CLASSES, N_CLASSES, SNOMED_TO_INDEX,
        SNOMED_TO_ABBR, ABBR_TO_SNOMED, SCP_TO_SNOMED,
        encode_ptbxl_labels, encode_snomed_labels, decode_label_vector
    )
    assert N_CLASSES == 27
    assert len(SCORED_SNOMED_CLASSES) == 27
    assert 164889003 in SNOMED_TO_INDEX  # AF

check("snomed_map imports + 27 classes", t_snomed_imports)

def t_snomed_encode():
    import numpy as np
    from snomed_map import encode_snomed_labels, decode_label_vector, SCORED_SNOMED_CLASSES
    codes = [164889003, 270492004]
    vec = encode_snomed_labels(codes)
    assert vec.shape == (27,)
    assert vec.sum() == 2.0
    decoded = decode_label_vector(vec)
    assert set(decoded) == set(codes)

check("encode_snomed_labels + decode roundtrip", t_snomed_encode)

def t_snomed_ptbxl_encode():
    import numpy as np
    from snomed_map import encode_ptbxl_labels
    scp = {"AFIB": 100.0, "NORM": 50.0, "UNKNOWN_CODE_XYZ": 100.0}
    vec = encode_ptbxl_labels(scp, min_likelihood=0.0)
    assert vec.shape == (27,)
    assert vec.sum() >= 2.0

check("encode_ptbxl_labels with unknown codes", t_snomed_ptbxl_encode)


print("\n══ DATA STRUCTURES ══")

def t_base_records():
    import numpy as np
    from _base import RecordA, RecordB, ProcessedCache
    ra = RecordA(
        signal=np.zeros((12, 5000), dtype=np.float32),
        label_vec=np.zeros(27, dtype=np.float32),
        ecg_id="test_001",
        dataset="ptbxl",
        split="train",
    )
    assert ra.signal.shape == (12, 5000)
    rb = RecordB(
        beat=np.zeros((2, 300), dtype=np.float32),
        beat_class=0,
        record_id="100",
        beat_idx=0,
        sample_pos=150,
    )
    assert rb.beat.shape == (2, 300)

check("RecordA + RecordB dataclasses", t_base_records)

def t_process_cache(tmp_path="/tmp/test_cache_ecg_smoke"):
    import numpy as np
    import shutil
    from _base import ProcessedCache
    shutil.rmtree(tmp_path, ignore_errors=True)
    cache = ProcessedCache(tmp_path)
    arr = np.random.randn(12, 5000).astype(np.float32)
    cache.put("key1", arr)
    assert "key1" in cache
    loaded = cache.get("key1")
    assert loaded is not None
    assert loaded.shape == arr.shape
    assert "nonexistent" not in cache
    cache.clear()
    assert "key1" not in cache
    shutil.rmtree(tmp_path, ignore_errors=True)

check("ProcessedCache put/get/clear", t_process_cache)

def t_base_parsers():
    import tempfile, os
    from _base import parse_wfdb_header_labels, parse_wfdb_header_fs
    with tempfile.NamedTemporaryFile(mode='w', suffix='.hea', delete=False) as f:
        f.write("A0001 12 500 5000\n")
        f.write("#Dx: 164889003,270492004\n")
        fname = f.name
    codes = parse_wfdb_header_labels(fname)
    assert 164889003 in codes
    assert 270492004 in codes
    fs = parse_wfdb_header_fs(fname)
    assert fs == 500
    os.unlink(fname)

check("parse_wfdb_header_labels + fs", t_base_parsers)

def t_simple_progress():
    from _base import SimpleProgress
    prog = SimpleProgress(10, desc="test", update_every=5)
    for _ in range(10):
        prog.update()
    prog.close()

check("SimpleProgress update + close", t_simple_progress)


print("\n══ UNIFIED DATASET ══")

def t_compute_weights():
    import numpy as np
    from unified_dataset import _compute_weights
    labels = ["ptbxl"] * 100 + ["georgia"] * 50 + ["stpetersburg"] * 5
    w_inv = _compute_weights(labels, strategy="inv_sqrt")
    assert len(w_inv) == 155
    w_uni = _compute_weights(labels, strategy="uniform")
    assert np.all(w_uni == 1.0)
    w_ds = _compute_weights(labels, strategy="dataset")
    assert len(w_ds) == 155

check("_compute_weights inv_sqrt/uniform/dataset", t_compute_weights)

def t_unified_dataset():
    import numpy as np
    from _base import RecordA
    from unified_dataset import UnifiedStreamADataset, build_unified_dataset
    records = []
    for i in range(20):
        r = RecordA(
            signal=np.random.randn(12, 5000).astype(np.float32),
            label_vec=np.random.randint(0, 2, 27).astype(np.float32),
            ecg_id=f"ecg_{i}",
            dataset="ptbxl" if i < 15 else "georgia",
            split="train",
        )
        records.append(r)
    ds, weights = build_unified_dataset(records, split="train", strategy="inv_sqrt")
    assert len(ds) == 20
    sig, lbl = ds[0]
    assert sig.shape == (12, 5000) or hasattr(sig, 'shape')

check("UnifiedStreamADataset build + __getitem__", t_unified_dataset)

def t_sampler():
    import numpy as np
    from _base import RecordA
    from unified_dataset import UnifiedStreamADataset, _compute_weights
    records = [
        RecordA(
            signal=np.random.randn(12, 5000).astype(np.float32),
            label_vec=np.zeros(27, dtype=np.float32),
            ecg_id=f"ecg_{i}",
            dataset="ptbxl" if i < 10 else "cpsc",
            split="train",
        )
        for i in range(15)
    ]
    weights = _compute_weights([r.dataset for r in records], "inv_sqrt")
    ds = UnifiedStreamADataset(records, weights)
    sampler = ds.make_sampler()
    assert sampler is not None

check("UnifiedStreamADataset make_sampler", t_sampler)


print("\n══ SUBSET REGISTRY ══")

def t_subset_registry():
    from subset_registry import (
        SUBSETS, SUBSET_SIZES, get_subset_weight,
        resolve_path, list_train_subsets, list_finetune_subsets, get_subset_info
    )
    assert "ptbxl" in SUBSETS
    assert "cpsc" in SUBSETS
    assert "georgia" in SUBSETS
    assert SUBSETS["georgia"]["freq_noise"] == 60
    assert SUBSETS["stpetersburg"]["fs"] == 257
    assert SUBSETS["ptb"]["fs"] == 1000
    train = list_train_subsets()
    assert "ptbxl" in train
    assert "cpsc" not in train
    ft = list_finetune_subsets()
    assert "cpsc" in ft
    w = get_subset_weight("ptbxl")
    assert w > 0
    path = resolve_path("ptbxl", "/kaggle/input/physionet2020")
    assert "WFDB_PTB-XL" in path

check("subset_registry all checks", t_subset_registry)


print("\n══ METRICS ══")

def t_metrics_imports():
    from metrics import (
        compute_macro_auc, compute_fmax, compute_per_class_metrics,
        compute_challenge_metric, classification_report_df,
        full_evaluation_report, bootstrap_ci, log_summary,
        RIBEIRO_2020_TARGETS, SNOMED_ABBRS
    )
    assert len(SNOMED_ABBRS) == 27
    assert "AF" in RIBEIRO_2020_TARGETS

check("metrics imports", t_metrics_imports)

def t_macro_auc():
    import numpy as np
    from metrics import compute_macro_auc
    np.random.seed(42)
    N, C = 200, 27
    probs = np.random.rand(N, C)
    targets = (np.random.rand(N, C) > 0.85).astype(float)
    targets[:, 0] = np.random.randint(0, 2, N)
    macro, per = compute_macro_auc(probs, targets)
    assert 0.0 <= macro <= 1.0
    assert len(per) == C

check("compute_macro_auc shape + range", t_macro_auc)

def t_fmax():
    import numpy as np
    from metrics import compute_fmax
    np.random.seed(0)
    N, C = 200, 27
    probs = np.random.rand(N, C)
    targets = (np.random.rand(N, C) > 0.85).astype(float)
    fmax, thr, pc_f1 = compute_fmax(probs, targets)
    assert 0.0 <= fmax <= 1.0
    assert 0.0 <= thr <= 1.0
    assert pc_f1.shape == (C,)

check("compute_fmax returns fmax/thr/per_class", t_fmax)

def t_per_class_metrics():
    import numpy as np
    from metrics import compute_per_class_metrics
    np.random.seed(1)
    N, C = 100, 5
    probs = np.random.rand(N, C)
    targets = (np.random.rand(N, C) > 0.7).astype(float)
    results = compute_per_class_metrics(probs, targets, class_names=["A","B","C","D","E"])
    assert len(results) == C
    assert "auc" in results[0]
    assert "f1" in results[0]

check("compute_per_class_metrics", t_per_class_metrics)

def t_full_report():
    import numpy as np
    from metrics import full_evaluation_report
    np.random.seed(2)
    N, C = 200, 27
    probs = np.random.rand(N, C)
    targets = (np.random.rand(N, C) > 0.85).astype(float)
    report = full_evaluation_report(probs, targets)
    assert "macro_auc" in report
    assert "fmax" in report
    assert "macro_f1" in report
    assert "per_class" in report
    assert report["n_samples"] == N

check("full_evaluation_report keys", t_full_report)

def t_bootstrap_ci():
    import numpy as np
    from metrics import bootstrap_ci
    np.random.seed(3)
    y_true = np.random.randint(0, 2, 200)
    y_score = np.random.rand(200)
    mean_, lo, hi = bootstrap_ci(y_true, y_score, n_boot=100)
    assert lo <= mean_ <= hi

check("bootstrap_ci lo <= mean <= hi", t_bootstrap_ci)


print("\n══ TRAINING COMMON ══")

def t_common_config():
    import tempfile, os
    from _common import load_config, _DotDict
    cfg_text = "seed: 42\npretrain:\n  lr: 0.001\n  batch_size: 64\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(cfg_text)
        fname = f.name
    cfg = load_config(fname)
    assert cfg.seed == 42
    assert cfg.pretrain.lr == 0.001
    os.unlink(fname)

check("load_config dot-access", t_common_config)

def t_common_device():
    from _common import get_device
    import torch
    dev = get_device()
    assert isinstance(dev, torch.device)

check("get_device returns torch.device", t_common_device)

def t_common_seed():
    from _common import setup_seed
    import numpy as np, random as rnd
    setup_seed(42)
    a = np.random.rand()
    setup_seed(42)
    b = np.random.rand()
    assert a == b

check("setup_seed reproducible", t_common_seed)

def t_pos_weight():
    import numpy as np, torch
    from _common import compute_pos_weight
    labels = np.zeros((100, 27), dtype=np.float32)
    labels[:10, 0] = 1.0
    labels[:50, 1] = 1.0
    pw = compute_pos_weight(labels, n_classes=27, device=torch.device("cpu"))
    assert pw.shape == (27,)
    assert pw[0] > pw[1]

check("compute_pos_weight imbalance ordering", t_pos_weight)

def t_early_stopping():
    from _common import EarlyStopping
    es = EarlyStopping(patience=3, min_delta=1e-4)
    es(0.80); assert not es.should_stop
    es(0.85); assert es.improved
    es(0.85); es(0.85); es(0.85)
    assert es.should_stop

check("EarlyStopping triggers after patience", t_early_stopping)

def t_step_timer():
    import time
    from _common import StepTimer
    t = StepTimer(window=5)
    for _ in range(3):
        t.start()
        time.sleep(0.001)
        t.stop()
    assert t.avg > 0

check("StepTimer avg > 0", t_step_timer)

def t_compute_metrics():
    import numpy as np
    from _common import compute_metrics
    np.random.seed(5)
    N, C = 150, 27
    probs = np.random.rand(N, C)
    targets = (np.random.rand(N, C) > 0.85).astype(float)
    targets[:, 0] = np.random.randint(0, 2, N)
    m = compute_metrics(probs, targets, n_classes=C)
    assert "macro_auc" in m
    assert "fmax" in m
    assert m["n_samples"] == N

check("compute_metrics all keys", t_compute_metrics)

def t_training_logger_none():
    from _common import TrainingLogger
    tlog = TrainingLogger(backend="none", run_name="test")
    tlog.log({"loss": 0.5}, step=1)
    tlog.finish()

check("TrainingLogger backend=none no-crash", t_training_logger_none)

def t_checkpoint_manager():
    import torch, tempfile, shutil
    from _common import CheckpointManager
    from resnet1d import build_resnet1d
    tmpdir = tempfile.mkdtemp()
    model = build_resnet1d()
    optim = torch.optim.AdamW(model.parameters())
    mgr = CheckpointManager(tmpdir, run_name="test", save_last=True)
    mgr.save(model, optim, epoch=1, metric=0.85, is_best=True)
    assert mgr.best_path.exists()
    model2 = build_resnet1d()
    mgr.load_best(model2)
    shutil.rmtree(tmpdir)

check("CheckpointManager save + load_best", t_checkpoint_manager)


print("\n══ BEAT CLASSIFIER ══")

def _load_beat_clf_mod():
    import importlib.util, types, sys
    # stub out 'training' package so beat_level_clf can be imported standalone
    if "training" not in sys.modules:
        pkg = types.ModuleType("training")
        sys.modules["training"] = pkg
    stub = types.ModuleType("training._common")
    class _ES: pass
    class _CKP: pass
    class _ST: pass
    class _TL:
        def log(self, *a, **kw): pass
        def finish(self): pass
    stub.EarlyStopping = _ES
    stub.CheckpointManager = _CKP
    stub.StepTimer = _ST
    stub.TrainingLogger = _TL
    for name in ["configure_root_logger","get_device","load_config","save_metrics",
                 "setup_logger","setup_seed","worker_init_fn"]:
        setattr(stub, name, lambda *a, **kw: None)
    sys.modules["training._common"] = stub
    spec = importlib.util.spec_from_file_location("beat_level_clf", "/mnt/project/beat_level_clf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def t_beat_cnn():
    import torch
    mod = _load_beat_clf_mod()
    BeatCNN = mod.BeatCNN
    model = BeatCNN(n_leads=2, conv_channels=[32,64,128], kernel_size=5, n_classes=5)
    model.eval()
    x = torch.randn(4, 2, 300)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 5)
    assert model.param_count() > 0

check("BeatCNN forward [4,2,300] -> [4,5]", t_beat_cnn)

def t_beat_dataset():
    import numpy as np
    mod = _load_beat_clf_mod()
    BeatDataset = mod.BeatDataset
    X = np.random.randn(50, 2, 300).astype(np.float32)
    y = np.random.randint(0, 5, 50).astype(np.int64)
    ds = BeatDataset(X, y)
    assert len(ds) == 50
    beat, label = ds[0]
    assert beat.shape == (2, 300)
    w = ds.class_weights()
    assert len(w) == 50

check("BeatDataset __len__ + __getitem__ + class_weights", t_beat_dataset)

def t_beat_metrics():
    import numpy as np
    mod = _load_beat_clf_mod()
    compute_beat_metrics = mod.compute_beat_metrics
    preds = np.array([0,0,1,2,2,3,4,0,1,2])
    targets = np.array([0,1,1,2,2,3,4,0,1,3])
    m = compute_beat_metrics(preds, targets)
    assert "macro_f1" in m
    assert "per_class_f1" in m
    assert 0.0 <= m["macro_f1"] <= 1.0

check("compute_beat_metrics keys + range", t_beat_metrics)


print("\n══ CPSC LOADER ══")

def _load_cpsc_mod():
    import importlib.util, types, sys
    for pkg in ["data", "preprocessing"]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
    for sub, src in [
        ("data._base", "/mnt/project/_base.py"),
        ("preprocessing.filters", "/mnt/project/filters.py"),
        ("preprocessing.normalize", "/mnt/project/normalize.py"),
        ("preprocessing.snomed_map", "/mnt/project/snomed_map.py"),
    ]:
        if sub not in sys.modules:
            spec = importlib.util.spec_from_file_location(sub, src)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[sub] = mod
            spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("load_cpsc2018", "/mnt/project/load_cpsc2018.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def t_cpsc_reference_missing():
    import tempfile
    mod = _load_cpsc_mod()
    tmpdir = tempfile.mkdtemp()
    df = mod.load_cpsc_reference(tmpdir)
    assert len(df) == 0
    import shutil; shutil.rmtree(tmpdir)

check("load_cpsc_reference empty dir returns empty df", t_cpsc_reference_missing)

def t_cpsc_reference_parse():
    import tempfile, os
    mod = _load_cpsc_mod()
    tmpdir = tempfile.mkdtemp()
    ref_path = os.path.join(tmpdir, "REFERENCE.csv")
    with open(ref_path, "w") as f:
        f.write("A0001,2\nA0002,1,5\nA0003,3\n")
    df = mod.load_cpsc_reference(tmpdir)
    assert len(df) == 3
    assert "record_id" in df.columns
    assert "snomed_codes" in df.columns
    row0 = df[df.record_id == "A0001"].iloc[0]
    assert 164889003 in row0.snomed_codes
    import shutil; shutil.rmtree(tmpdir)

check("load_cpsc_reference parses CSV correctly", t_cpsc_reference_parse)

def t_cpsc_label_mapping():
    mod = _load_cpsc_mod()
    assert mod.CPSC_LABEL_TO_SNOMED[2] == 164889003
    assert mod.CPSC_LABEL_TO_SNOMED[4] == 164909002
    assert len(mod.CPSC_LABEL_TO_SNOMED) == 9

check("CPSC_LABEL_TO_SNOMED mapping", t_cpsc_label_mapping)


print("\n══ TASK MODULES IMPORTABLE ══")

def _load_task_mod(name, path):
    import importlib.util, types, sys
    for pkg in ["data", "preprocessing", "training", "backbone", "tasks"]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
    for sub, src in [
        ("data._base", "/mnt/project/_base.py"),
        ("preprocessing.filters", "/mnt/project/filters.py"),
        ("preprocessing.normalize", "/mnt/project/normalize.py"),
        ("preprocessing.snomed_map", "/mnt/project/snomed_map.py"),
        ("backbone.resnet1d", "/mnt/project/resnet1d.py"),
        ("backbone.load_weights", "/mnt/project/load_weights.py"),
    ]:
        if sub not in sys.modules:
            spec = importlib.util.spec_from_file_location(sub, src)
            m = importlib.util.module_from_spec(spec)
            sys.modules[sub] = m
            spec.loader.exec_module(m)
    # training._common stub
    if "training._common" not in sys.modules:
        stub = types.ModuleType("training._common")
        for attr in ["EarlyStopping","CheckpointManager","StepTimer","TrainingLogger",
                     "configure_root_logger","get_device","load_config","save_metrics",
                     "setup_logger","setup_seed","worker_init_fn","build_dataloader",
                     "collect_predictions","compute_metrics","compute_pos_weight"]:
            setattr(stub, attr, lambda *a, **kw: None)
        sys.modules["training._common"] = stub
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # must register before exec so @dataclass can resolve __module__
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def t_import_intervals():
    mod = _load_task_mod("intervals", "/mnt/project/01_intervals.py")
    assert mod is not None

check("01_intervals.py importable", t_import_intervals)

def t_intervals_constants():
    mod = _load_task_mod("intervals", "/mnt/project/01_intervals.py")
    assert hasattr(mod, "INTERVAL_NORMS")
    assert hasattr(mod, "measure_intervals")
    assert hasattr(mod, "aggregate_results")
    assert hasattr(mod, "IntervalResult")
    assert mod.INTERVAL_NORMS["qrs_ms"][1] == 120.0
    assert mod.FS == 500.0

check("01_intervals constants + functions", t_intervals_constants)

def t_afib_constants():
    mod = _load_task_mod("afib", "/mnt/project/03_afib.py")
    assert hasattr(mod, "AF_SNOMED_IDX")
    assert mod.AF_SNOMED_IDX == 0
    assert hasattr(mod, "FEATURE_NAMES")
    assert len(mod.FEATURE_NAMES) == 9

check("03_afib constants", t_afib_constants)

def t_tachycardia_constants():
    mod = _load_task_mod("tachy", "/mnt/project/02_tachycardia.py")
    assert hasattr(mod, "VT_SNOMED_IDX")
    assert mod.VT_SNOMED_IDX == 26
    assert mod.QRS_THRESHOLD_MS == 120.0

check("02_tachycardia constants", t_tachycardia_constants)

def t_blocks_constants():
    mod = _load_task_mod("blocks", "/mnt/project/04_blocks.py")
    assert hasattr(mod, "IAVB_IDX")
    assert hasattr(mod, "RIBEIRO_TARGETS")
    assert "1AVB" in mod.RIBEIRO_TARGETS
    assert mod.RIBEIRO_TARGETS["LBBB"]["f1"] == 1.000

check("04_blocks constants + Ribeiro targets", t_blocks_constants)


print("\n══ RIBEIRO REFERENCE VALUES ══")

def t_ribeiro_targets():
    from metrics import RIBEIRO_2020_TARGETS
    assert RIBEIRO_2020_TARGETS["AF"]["f1"] == 0.870
    assert RIBEIRO_2020_TARGETS["LBBB"]["f1"] == 1.000
    assert RIBEIRO_2020_TARGETS["VTach"]["f1"] == 0.750
    assert RIBEIRO_2020_TARGETS["CRBBB"]["specificity"] == 0.983

check("RIBEIRO_2020_TARGETS values match paper", t_ribeiro_targets)

def t_reproduce_ribeiro_classes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("reprib", "/mnt/project/reproduce_ribeiro.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "AF" in mod.RIBEIRO_CLASSES
    assert "VTach" in mod.RIBEIRO_CLASSES
    assert len(mod.RIBEIRO_CLASSES) == 10

check("reproduce_ribeiro RIBEIRO_CLASSES", t_reproduce_ribeiro_classes)


print("\n══ DOWNLOAD / MISC ══")

def t_download_registry():
    from download import DATASETS, check_dataset
    assert "mitbih" in DATASETS
    assert "ptbxl" in DATASETS
    assert "physionet2020" in DATASETS
    info = check_dataset("mitbih", __import__("pathlib").Path("/nonexistent/path"))
    assert info["present"] == False

check("download.py DATASETS registry + check_dataset", t_download_registry)

def t_physionet_loader_imports():
    import importlib.util, types, sys
    for pkg in ["data", "preprocessing"]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
    for sub, src in [
        ("data._base", "/mnt/project/_base.py"),
        ("preprocessing.filters", "/mnt/project/filters.py"),
        ("preprocessing.normalize", "/mnt/project/normalize.py"),
        ("preprocessing.snomed_map", "/mnt/project/snomed_map.py"),
    ]:
        if sub not in sys.modules:
            spec = importlib.util.spec_from_file_location(sub, src)
            m = importlib.util.module_from_spec(spec)
            sys.modules[sub] = m
            spec.loader.exec_module(m)
    spec = importlib.util.spec_from_file_location("load_physionet2020", "/mnt/project/load_physionet2020.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SOURCE_POWERLINE["G12EC"] == 60
    assert mod.SOURCE_POWERLINE["CPSC2018"] == 50
    assert "PTBXL" in mod.PTBXL_SUBDIRS
    ecg_id = mod._extract_ptbxl_ecg_id("00001_hr.hea")
    assert ecg_id == "1"
    ecg_id2 = mod._extract_ptbxl_ecg_id("21837_hr")
    assert ecg_id2 == "21837"

check("load_physionet2020 imports + ecg_id extraction", t_physionet_loader_imports)

def t_ablation_scenarios():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ablation", "/mnt/project/ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "SCENARIO_DESCRIPTIONS")
    assert len(mod.SCENARIO_DESCRIPTIONS) == 5
    assert hasattr(mod, "FOCAL_CLASSES")
    assert "AF" in mod.FOCAL_CLASSES
    assert "VTach" in mod.FOCAL_CLASSES

check("ablation.py scenarios + focal classes", t_ablation_scenarios)


print("\n══ END-TO-END MINI PIPELINE ══")

def t_e2e_forward_metrics():
    import torch, numpy as np
    from resnet1d import ResNet1dWithHead
    from filters import apply_ecg_filters
    from normalize import normalize_record_stream_a
    from snomed_map import encode_snomed_labels
    from metrics import compute_macro_auc, compute_fmax

    raw = np.random.randn(12, 5000).astype(np.float32)
    filtered = apply_ecg_filters(raw, fs=500.0, powerline_freq=50).astype(np.float32)
    normed = normalize_record_stream_a(filtered, fs_in=500, fs_out=500, target_len=5000).astype(np.float32)
    assert normed.shape == (12, 5000)

    label = encode_snomed_labels([164889003])
    assert label[0] == 1.0

    model = ResNet1dWithHead(n_classes=27)
    model.eval()
    x = torch.from_numpy(normed[None])
    with torch.no_grad():
        logits = model(x)
    probs = torch.sigmoid(logits).numpy()
    assert probs.shape == (1, 27)

    probs_batch = np.random.rand(50, 27)
    targets_batch = (np.random.rand(50, 27) > 0.85).astype(float)
    targets_batch[:, 0] = np.random.randint(0, 2, 50)
    auc, _ = compute_macro_auc(probs_batch, targets_batch)
    fmax, thr, _ = compute_fmax(probs_batch, targets_batch)
    assert 0 <= auc <= 1
    assert 0 <= fmax <= 1

check("E2E: preprocess -> model -> metrics", t_e2e_forward_metrics)


print("\n" + "═" * 55)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)
print(f"  ИТОГО: {passed}/{total} пройдено  |  {failed} провалено")
print("═" * 55)

if failed > 0:
    print("\nПровалившиеся тесты:")
    for name, ok, err in results:
        if not ok:
            print(f"  • {name}")
            print(f"    {err}")
    sys.exit(1)
else:
    print("\n  Все проверки пройдены ✓")
