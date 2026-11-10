# ClassWatch Live: classroom demo on a laptop webcam

Point your laptop webcam at the class and run one command. ClassWatch Live shows, in real time:

| On screen | How it is worked out |
|---|---|
| **People in view**, each with an anonymous number (#3) | YOLO11 pose model + ByteTrack tracking. Numbers are random and forgotten when the program closes. |
| **Hands raised** now, and how many times this session | A wrist above the head, held for a few frames |
| **Heads: front / turned / down / away**, and **% facing front** | Where the nose sits between the eyes; how close the nose is to the shoulder line |
| **Standing / sitting** | Hip, knee and ankle positions (shown only when legs are visible) |
| **Phones in view** (optional) | YOLO11 object detection, COCO class "cell phone" |
| **Door in / out** (optional) | Anonymous tracks crossing a line you draw over the door |
| **60-second trend** of % facing front | Averaged once per second |

**Privacy by default.** Faces are pixelated on screen. No video or image is saved. Nobody is identified. `--log` saves only class-level counts per second, with no IDs and no images.

**Read the cues as hints, not judgements.** "Head down" can mean writing, reading or sleeping. "Turned" can mean talking to a neighbour or looking at a side board. Telling these apart is the open research problem in the main project.

## Windows: one-click app (no Python needed)

1. Open the [Releases page](https://github.com/Zubair-DLCS/ClassWatch/releases/latest) and download **`ClassWatch-Setup-….exe`**.
2. Run it. Windows may say *"Windows protected your PC"* because the app is not code-signed: click **More info → Run anyway**. No admin rights are needed.
3. Click the **ClassWatch** icon on your desktop. A small start window opens: choose the camera, tick the options you want, and press **Start ClassWatch**.

The models are included, so it works without internet. Counts you choose to save, and any error report, go to `Documents\ClassWatch`.
A portable version (`ClassWatch-portable-….zip`) runs without installing: unzip it and double-click `ClassWatch.exe`.

The app is built and tested automatically on a Windows machine by GitHub Actions ([workflow](../.github/workflows/build-windows.yml)). To publish a new version, open Actions → *Build Windows app* → *Run workflow* and enter a version such as `1.0.1` (or push a tag `v1.0.1`).

## Run it from source (Windows, macOS or Linux)

You need Python 3.9 or newer. In a terminal, from the repository folder:

```bash
pip install -r live/requirements.txt
python live/app.py                # start window
python live/classwatch_live.py    # straight to the camera
```

The first run downloads two small models (about 6 MB each). After that it works offline. On Windows you can also double-click `live/start_classwatch.bat`.

Useful options:

```bash
python live/classwatch_live.py --phones                         # also look for phones
python live/classwatch_live.py --door 0.85,0.1,0.85,0.95        # door line: x1,y1,x2,y2 as 0..1 fractions of the image
python live/classwatch_live.py --source 1                       # external USB camera
python live/classwatch_live.py --source lecture.mp4             # a recorded video
python live/classwatch_live.py --model yolo11s-pose.pt          # more accurate, slower
python live/classwatch_live.py --log class_counts.csv           # class-level counts per second
```

Keys while running: **q** quit · **b** blur on/off · **k** skeleton on/off · **r** reset counters · **f** fullscreen.

## Tips for a classroom demo

- Put the laptop on the teacher's desk, raised a little (on books), facing the class. A USB webcam on a tripod (`--source 1`) sees more of the room.
- A small CPU model (`yolo11n-pose`) runs at about 10–20 frames per second on an ordinary laptop. Students at the back of a big room may be too small to detect; try `--imgsz 960` or the `s` model.
- Ask a few students to raise hands, turn around, look down and stand up. The panel updates within a second.
- Tell the class before you start that the camera is on, faces are blurred and nothing is recorded. That conversation is part of the lesson.

## Why it is not run in Colab

Colab runs on a server far away. Every webcam frame must travel from your browser to the server and back, which is slow and breaks easily. This demo runs on your own laptop, so the video never leaves the room.

## Licence note

The detection and pose models come from [Ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0.
