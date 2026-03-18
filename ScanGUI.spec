# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['scan_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('Data.txt', '.')],
    hiddenimports=['pyvisa_py', 'pyvisa_py.highlevel', 'pyvisa_py.sessions', 'pyvisa_py.usb', 'pyvisa_py.protocols.usbtmc', 'serial.tools.list_ports'],
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
    name='ScanGUI',
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
    name='ScanGUI.app',
    icon=None,
    bundle_identifier=None,
)
