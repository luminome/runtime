pyinstaller --windowed --noconfirm --debug --log-level=DEBUG runtime.spec
cp -f Info.plist dist/x29.app/Contents/Info.plist
echo 'process complete'