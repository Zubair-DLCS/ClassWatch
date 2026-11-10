"""ClassWatch desktop app: a small start window in front of ClassWatch Live.

Double-click the ClassWatch icon -> choose the camera and options -> Start.
Run with command-line options (e.g. `ClassWatch.exe --source 0 --phones`) to
skip the window and start directly.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# A windowed .exe has no console: give libraries somewhere harmless to print.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APP = "ClassWatch"
DOORS = {
    "No door counting": None,
    "Door on the right edge": "0.9,0.05,0.9,0.95",
    "Door on the left edge": "0.1,0.05,0.1,0.95",
}


def resource(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(base, name), os.path.join(base, "..", "packaging", name)):
        if os.path.exists(p):
            return p
    return ""


def counts_file() -> str:
    folder = Path.home() / "Documents" / APP
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / time.strftime("class_counts_%Y-%m-%d_%H%M.csv"))


def run_live(args: list[str]) -> str | None:
    """Run ClassWatch Live; return an error message, or None if it ended normally.
    Problems are also written to Documents/ClassWatch/last_error.txt."""
    import traceback
    err, detail = None, ""
    try:
        import classwatch_live
        classwatch_live.main(args)
    except SystemExit as e:
        if e.code not in (None, 0):
            err = detail = str(e.code)
    except Exception as e:  # show the problem instead of closing silently
        err, detail = f"{type(e).__name__}: {e}", traceback.format_exc()
    if err:
        try:
            folder = Path.home() / "Documents" / APP
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "last_error.txt").write_text(f"{time.ctime()}\nargs: {args}\n\n{detail}\n")
        except OSError:
            pass
    return err


def launcher(auto_close_ms: int | None = None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title(APP)
    root.resizable(False, False)
    ico = resource("classwatch.ico")
    if ico:
        try:
            root.iconbitmap(ico)
        except tk.TclError:
            pass

    pad = {"padx": 14, "pady": 4}
    tk.Label(root, text="ClassWatch Live", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(14, 0))
    tk.Label(root, text="See how the class is doing, without recording anyone.", fg="#555").grid(row=1, column=0, columnspan=2, sticky="w", **pad)

    tk.Label(root, text="Camera").grid(row=2, column=0, sticky="w", **pad)
    cam = ttk.Combobox(root, state="readonly", width=34, values=[
        "Laptop camera (0)", "USB / second camera (1)", "Third camera (2)", "A video file..."])
    cam.current(0)
    cam.grid(row=2, column=1, sticky="w", **pad)
    video_path = {"path": None}

    def on_cam(_):
        if cam.get().startswith("A video file"):
            p = filedialog.askopenfilename(title="Choose a video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
            if p:
                video_path["path"] = p
                cam.set(f"Video: {os.path.basename(p)}")
            else:
                cam.current(0)
    cam.bind("<<ComboboxSelected>>", on_cam)

    blur = tk.BooleanVar(value=True)
    phones = tk.BooleanVar(value=False)
    log = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Blur faces (recommended)", variable=blur).grid(row=3, column=1, sticky="w", **pad)
    tk.Checkbutton(root, text="Look for mobile phones (a little slower)", variable=phones).grid(row=4, column=1, sticky="w", **pad)
    tk.Label(root, text="Door").grid(row=5, column=0, sticky="w", **pad)
    door = ttk.Combobox(root, state="readonly", width=34, values=list(DOORS))
    door.current(0)
    door.grid(row=5, column=1, sticky="w", **pad)
    tk.Checkbutton(root, text="Save class counts per second (CSV, no images)", variable=log).grid(row=6, column=1, sticky="w", **pad)

    tk.Label(root, justify="left", fg="#2e7d32", text=(
        "Nothing is recorded. Students get random numbers, not names.\n"
        "Keys: q quit · b blur · k skeleton · r reset · f fullscreen")).grid(row=7, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 4))

    status = tk.Label(root, text="", fg="#555")
    status.grid(row=9, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 12))

    def start():
        args = []
        c = cam.get()
        if c.startswith("Video:") and video_path["path"]:
            args += ["--source", video_path["path"]]
        else:
            args += ["--source", c[c.rfind("(") + 1:c.rfind(")")] if "(" in c else "0"]
        if not blur.get():
            if not messagebox.askyesno(APP, "Show faces unblurred?\n\nOnly do this if everyone in view has agreed."):
                return
            args.append("--no-blur")
        if phones.get():
            args.append("--phones")
        if DOORS[door.get()]:
            args += ["--door", DOORS[door.get()]]
        csv_path = None
        if log.get():
            csv_path = counts_file()
            args += ["--log", csv_path]
        status.config(text="Starting the camera and loading the model...")
        btn.config(state="disabled")
        root.update()
        root.withdraw()
        err = run_live(args)
        root.deiconify()
        btn.config(state="normal")
        if err:
            status.config(text="Stopped with a problem.")
            messagebox.showerror(APP, err)
        else:
            status.config(text=f"Session ended. Counts saved to {csv_path}" if csv_path else "Session ended.")

    btn = tk.Button(root, text="Start ClassWatch", font=("Segoe UI", 12, "bold"), bg="#2f6db5", fg="white",
                    activebackground="#24578f", activeforeground="white", relief="flat", padx=18, pady=6, command=start)
    btn.grid(row=8, column=0, columnspan=2, pady=(6, 6))
    root.bind("<Return>", lambda _e: start())
    if auto_close_ms:  # used by the automatic build test: screenshot the window, then close
        def snap():
            try:
                from PIL import ImageGrab
                root.update()
                x, y = root.winfo_rootx(), root.winfo_rooty()
                w, h = root.winfo_width(), root.winfo_height()
                folder = Path.home() / "Documents" / APP
                folder.mkdir(parents=True, exist_ok=True)
                ImageGrab.grab(bbox=(x - 8, y - 32, x + w + 8, y + h + 8)).save(folder / "start_window.png")
            except Exception:
                pass
        root.after(auto_close_ms - 500, snap)
        root.after(auto_close_ms, root.destroy)
    root.mainloop()


def main():
    if sys.argv[1:] == ["--check-window"]:  # build test: open the start window, close it, report
        launcher(auto_close_ms=1500)
        (Path.home() / "Documents" / APP).mkdir(parents=True, exist_ok=True)
        (Path.home() / "Documents" / APP / "window_check_ok.txt").write_text("ok")
        return
    if len(sys.argv) > 1:  # direct start with options, no window
        err = run_live(sys.argv[1:])
        if err:
            print(err, file=sys.stderr)
            sys.exit(1)
        return
    launcher()


if __name__ == "__main__":
    main()
