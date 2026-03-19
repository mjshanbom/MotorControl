# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['scope_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('Data.txt', '.')],
    hiddenimports=[
        'pyvisa_py',
        'pyvisa_py.highlevel',
        'pyvisa_py.sessions',
        'pyvisa_py.sessions.serial_session',
        'pyvisa_py.sessions.tcpip_session',
        'pyvisa_py.sessions.usb_session',
        'pyvisa_py.usb',
        'pyvisa_py.protocols.usbtmc',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends._backend_tk',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ScopeGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='ScopeGUI.app',
    icon=None,
    bundle_identifier=None,
)
