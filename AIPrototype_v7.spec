# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('ikon.ico', '.'), ('sprak.py', '.'), ('retrieval_deep.py', '.')]
binaries = []
hiddenimports = ['sprak', 'retrieval_deep']

for _pkg in ('llama_cpp', 'PIL', 'certifi', 'requests', 'charset_normalizer',
             'trafilatura', 'numpy', 'lxml', 'courlan', 'htmldate', 'justext'):
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception:
        pass

a = Analysis(
    ['AIPrototype.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rth_stdio.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# ONEFILE: allt (kod + DLL:er) i EN exe. upx=False for att inte riskera
# att korrupta CUDA-DLL:er. Modellerna bundlas INTE - de lases fran
# models/ bredvid exe:n (BASKATALOG = exe-katalogen nar frozen).
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AIPrototype_v7',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ikon.ico',
)
