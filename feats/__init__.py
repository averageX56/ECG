"""Аутентичный пакет ручных ЭКГ-фич команды Prna / S. Goodfellow (PhysioNet 2017/2020).

Извлечение фич — в точности как в исходном коде (features.py + три группы
статистик + utils). При импорте применяется тонкий слой совместимости со
старыми версиями библиотек, под которые писался код: устаревшие псевдонимы
numpy (np.int/np.float/np.trapz), pd.DataFrame.append и сигнатура
pyentrp.permutation_entropy(m=...). Без этого код падает на новых версиях.
"""
from __future__ import annotations


def _apply_compat():
    import numpy as np
    import pandas as pd

    for name, typ in (('int', int), ('float', float), ('bool', bool)):
        if not hasattr(np, name):
            setattr(np, name, typ)
    if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
        np.trapz = np.trapezoid
    # удалённые в numpy 2.0 псевдонимы бесконечности/NaN
    for name, val in (('Inf', np.inf), ('Infinity', np.inf), ('infty', np.inf),
                      ('PINF', np.inf), ('NINF', -np.inf),
                      ('NaN', np.nan), ('NAN', np.nan)):
        if not hasattr(np, name):
            setattr(np, name, val)

    if not hasattr(pd.DataFrame, 'append'):
        def _append(self, other, ignore_index=False, **kwargs):
            if isinstance(other, pd.Series):
                other = other.to_frame().T
            return pd.concat([self, other], ignore_index=ignore_index)
        pd.DataFrame.append = _append

    try:
        from pyentrp import entropy as ent
        if not getattr(ent, '_compat_pe', False):
            _orig = ent.permutation_entropy

            def permutation_entropy(time_series, order=3, delay=1, normalize=False, m=None):
                if m is not None:
                    order = m
                return _orig(time_series, order=order, delay=delay, normalize=normalize)

            ent.permutation_entropy = permutation_entropy
            ent._compat_pe = True
    except Exception:
        pass


_apply_compat()
