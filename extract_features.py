"""
extract_features.py
Отдельный пайплайн извлечения ручных фич (решение команды Prna) и запись в CSV.

Считает фичи В ТОЧНОСТИ как в архиве — через feats.Features (три группы
статистик: full_waveform / heart_rate_variability / template). Обычный
последовательный проход по записям (без мультипроцессинга), с прогресс-баром.
Результат — CSV с колонкой record_id и всеми ~300 фичами; его читает
пайплайн 3 (train_ctn.py --features features.csv).

Пример:
  python extract_features.py --data-root data --out features.csv --ch-idx 1
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

import feats  # применяет слой совместимости
from feats.features import Features
from ecg_data import build_record_table, read_record, CLASSES

FEATURE_GROUPS = ['full_waveform_statistics',
                  'heart_rate_variability_statistics',
                  'template_statistics']


def extract_one(signal, fs, ch_idx=1, filter_bandwidth=(3.0, 45.0),
                template_before=0.25, template_after=0.4, seed=42) -> dict:
    """Фичи одной записи по отведению ch_idx -> dict {имя: значение}.

    seed фиксирует numpy RNG -> детерминированный KMeans в HRV-статистиках.
    """
    ch = min(ch_idx, signal.shape[0] - 1)
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ef = Features(data=np.asarray(signal[ch], dtype=float), fs=int(fs),
                      feature_groups=FEATURE_GROUPS)
        ef.calculate_features(filter_bandwidth=list(filter_bandwidth), show=False,
                              channel=0, normalize=True, polarity_check=True,
                              template_before=template_before, template_after=template_after)
        row = ef.get_features().iloc[0].to_dict()
    row.pop('file_name', None)
    return row


def extract_dataset(df: pd.DataFrame, ch_idx=1, filter_bandwidth=(3.0, 45.0),
                    fs_out=500, show_progress=True) -> pd.DataFrame:
    """Последовательно считает фичи по всем записям df -> DataFrame [record_id, <фичи>]."""
    it = df.itertuples(index=False)
    if show_progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(df), desc='Извлечение фич')
        except Exception:
            pass
    rows, n_fail = [], 0
    for r in it:
        rec_dict = {'record_id': r.record_id}
        try:
            rec = read_record(r.filepath, fs_out=fs_out)
            rec_dict.update(extract_one(rec.signal, rec.fs, ch_idx=ch_idx,
                                        filter_bandwidth=filter_bandwidth))
        except Exception:
            n_fail += 1
        rows.append(rec_dict)
    if n_fail:
        print(f'extract_dataset: {n_fail}/{len(df)} записей не дали фич (строка почти пустая)')
    out = pd.DataFrame(rows)
    # числовые фичи: inf -> nan (заполнение средним делает уже обучение)
    num = out.columns.drop('record_id')
    out[num] = out[num].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return out


def load_features(csv_path) -> dict:
    """Читает CSV фич -> {record_id: np.ndarray} + список имён фич.

    Возвращает (features_dict, feature_names). Порядок фич фиксирован именами колонок.
    """
    df = pd.read_csv(csv_path)
    df['record_id'] = df['record_id'].astype(str)
    names = [c for c in df.columns if c != 'record_id']
    vals = df[names].to_numpy(dtype=np.float64)
    features = {rid: vals[i] for i, rid in enumerate(df['record_id'].tolist())}
    return features, names


# --------------------------------------------------------------------------
# Отбор/снижение размерности фич (считается по train, без top_feats.npy)
# --------------------------------------------------------------------------
def _train_matrix(features, train_df):
    """Матрица фич train (inf/NaN -> среднее по столбцу). Возвращает (ids, X, col_mean)."""
    ids = [r for r in train_df['record_id'].tolist() if r in features]
    X = np.vstack([features[i] for i in ids]).astype(np.float64)
    X[np.isinf(X)] = np.nan
    col_mean = np.nan_to_num(np.nanmean(X, axis=0))
    X = np.where(np.isnan(X), col_mean, X)
    return ids, X, col_mean


def fit_selector(features, src_names, train_df, method='rf', nb_feats=64, seed=42) -> dict:
    """Подбирает отбор фич ПО TRAIN и возвращает сериализуемый селектор (dict).

    method:
      'rf'          — топ-N по важности RandomForest (как в оригинальном top_feats);
      'variance'    — топ-N по дисперсии;
      'mutual_info' — топ-N по сумме взаимной информации с классами;
      'pca'         — N главных компонент (StandardScaler -> PCA).
    Селектор применяется apply_selector() и сохраняется в чекпоинт.
    """
    ids, X, col_mean = _train_matrix(features, train_df)
    nb = int(min(nb_feats, X.shape[1]))

    if method == 'pca':
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        sc = StandardScaler().fit(X)
        pca = PCA(n_components=nb, random_state=seed).fit(sc.transform(X))
        return {'method': 'pca', 'src_names': list(src_names), 'col_mean': col_mean.tolist(),
                'scaler_mean': sc.mean_.tolist(), 'scaler_scale': sc.scale_.tolist(),
                'pca_mean': pca.mean_.tolist(), 'pca_components': pca.components_.tolist(),
                'names': [f'pca_{i}' for i in range(pca.n_components_)]}

    if method == 'variance':
        score = np.var(X, axis=0)
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_classif
        Y = train_df.set_index('record_id').loc[ids][CLASSES].to_numpy()
        score = np.zeros(X.shape[1])
        for k in range(Y.shape[1]):
            if 0 < Y[:, k].sum() < len(Y):
                score += mutual_info_classif(X, Y[:, k], random_state=seed)
    else:  # 'rf'
        from sklearn.ensemble import RandomForestClassifier
        Y = train_df.set_index('record_id').loc[ids][CLASSES].to_numpy()
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=seed)
        rf.fit(X, Y)
        score = rf.feature_importances_

    idx = np.sort(np.argsort(score)[::-1][:nb])
    return {'method': 'subset', 'src_names': list(src_names), 'col_mean': col_mean.tolist(),
            'indices': idx.tolist(), 'names': [src_names[i] for i in idx]}


def apply_selector(features: dict, selector: dict) -> dict:
    """Применяет селектор (из fit_selector) к {record_id: вектор} -> новый dict."""
    if selector is None:
        return features
    if selector['method'] == 'pca':
        col_mean = np.asarray(selector['col_mean'])
        sc_mean = np.asarray(selector['scaler_mean'])
        sc_scale = np.where(np.asarray(selector['scaler_scale']) == 0, 1.0, selector['scaler_scale'])
        pca_mean = np.asarray(selector['pca_mean'])
        comp = np.asarray(selector['pca_components'])

        def tf(v):
            v = np.asarray(v, dtype=np.float64)
            v = np.where(np.isnan(v) | np.isinf(v), col_mean, v)
            return comp @ ((v - sc_mean) / sc_scale - pca_mean)
        return {rid: tf(vec) for rid, vec in features.items()}

    idx = np.asarray(selector['indices'])
    return {rid: np.asarray(vec, dtype=np.float64)[idx] for rid, vec in features.items()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Извлечение ручных фич Prna -> CSV')
    p.add_argument('--data-root', default='data')
    p.add_argument('--out', default='features.csv')
    p.add_argument('--ch-idx', type=int, default=1, help='Отведение (1 = II)')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--lo', type=float, default=3.0, help='Нижняя граница полосы фильтра')
    p.add_argument('--hi', type=float, default=45.0, help='Верхняя граница полосы фильтра')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = build_record_table(args.data_root, limit=args.limit)
    print(f'Записей: {len(df)}. Считаю фичи по отведению {args.ch_idx}...')
    feats_df = extract_dataset(df, ch_idx=args.ch_idx, filter_bandwidth=(args.lo, args.hi))
    feats_df.to_csv(args.out, index=False)
    print(f'Готово: {feats_df.shape[0]} записей x {feats_df.shape[1]-1} фич -> {args.out}')


if __name__ == '__main__':
    main()
