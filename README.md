# finger-scanning-camara

## Run

From the `src` directory, start the application with:

```powershell
python main.py
```

Press `Q` in the camera window to exit.

## Source structure

- `src/main.py` - application entry point.
- `src/finger_counter.py` - coordinates the camera, tracker, and UI.
- `src/camera.py` - camera setup, mirroring, and the punk CRT/VHS filter.
- `src/hand_tracking.py` - MediaPipe setup, hand analysis, and landmark drawing.
- `src/ui.py` - neon stencil text, HUD panel, paint drips, and punk decorators.
