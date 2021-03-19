# -*- mode: python -*-

block_cipher = None

a = Analysis(['Runtime.py'],
             pathex=['/Users/sac/Sites/runtime-2016-rev'],
             datas=[ ('pf_tempesta_seven.ttf', '.'),('untitled.obj', '.'),('untitled.mtl', '.'),('untitled3.obj', '.'),('untitled3.mtl', '.'),('untitled-sub.obj', '.'),('untitled-sub.mtl', '.'),('runtime-variables.plist', '.'),('Unknown.json', '.'),('audio/*', 'audio') ],
             binaries=None,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='runtime',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='runtime')
app = BUNDLE(coll,
             name='x29.app',
             icon='icons.icns',
             bundle_identifier='com.luminome.x29program')
