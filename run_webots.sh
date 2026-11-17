#!/bin/zsh
# Launch Webots R2025a on this Mac with the connect() fix.
# Why: on this macOS version Qt 6 reports refused TCP connections as "connected",
# so Webots believes every port is busy and refuses to start. libconnect_fix.dylib
# (source: ~/Applications/webots_fix/connect_fix.c) corrects the connect() result.
# Usage:  ./run_webots.sh worlds/cnn_object_world.wbt
export PATH="$HOME/.pyenv/versions/3.9.13/bin:$PATH"
export DYLD_INSERT_LIBRARIES="$HOME/Applications/webots_fix/libconnect_fix.dylib"
exec "$HOME/Applications/Webots-R2025a.app/Contents/MacOS/webots" "$@"
