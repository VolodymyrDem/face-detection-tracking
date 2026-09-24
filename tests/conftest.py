import os

# Qt widgets in tests render off-screen, without opening real windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
