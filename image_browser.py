"""
GUI Image Browser
-----------------
Browses GIF image files on the filesystem using a tkinter interface.
Modelled after the simple text-file browser pattern: open a file dialog,
load the selected image, and display it in the main window.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class ImageBrowser(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Browser")
        self.resizable(True, True)

        self._current_image = None   # keep a reference so GC doesn't collect it
        self._current_path = None
        self._history: list[str] = []
        self._history_index = -1

        self._build_menu()
        self._build_toolbar()
        self._build_canvas()
        self._build_statusbar()

        self.minsize(500, 400)
        self._update_nav_buttons()

    # ------------------------------------------------------------------ build

    def _build_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open…", accelerator="Ctrl+O", command=self._open)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="Ctrl+Q", command=self.destroy)

        view_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Fit to Window", command=self._fit_to_window)
        view_menu.add_command(label="Actual Size",   command=self._actual_size)

        self.bind_all("<Control-o>", lambda _e: self._open())
        self.bind_all("<Control-q>", lambda _e: self.destroy())

    def _build_toolbar(self):
        bar = ttk.Frame(self, relief="raised", padding=2)
        bar.pack(side="top", fill="x")

        self._btn_back = ttk.Button(bar, text="◀ Back",    command=self._go_back,    width=8)
        self._btn_fwd  = ttk.Button(bar, text="Forward ▶", command=self._go_forward, width=10)
        btn_open       = ttk.Button(bar, text="Open…",     command=self._open,       width=8)

        self._btn_back.pack(side="left", padx=2, pady=2)
        self._btn_fwd .pack(side="left", padx=2, pady=2)
        btn_open      .pack(side="left", padx=2, pady=2)

        self._path_var = tk.StringVar(value="No file loaded")
        path_label = ttk.Label(bar, textvariable=self._path_var,
                               anchor="w", relief="sunken", padding=(4, 2))
        path_label.pack(side="left", fill="x", expand=True, padx=(6, 2))

    def _build_canvas(self):
        frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        frame.pack(fill="both", expand=True, padx=4, pady=4)

        self._canvas = tk.Canvas(frame, bg="#2b2b2b", cursor="crosshair")
        h_scroll = ttk.Scrollbar(frame, orient="horizontal",
                                  command=self._canvas.xview)
        v_scroll = ttk.Scrollbar(frame, orient="vertical",
                                  command=self._canvas.yview)

        self._canvas.configure(xscrollcommand=h_scroll.set,
                               yscrollcommand=v_scroll.set)

        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right",  fill="y")
        self._canvas.pack(fill="both", expand=True)

        # Show a welcome message on the blank canvas
        self._canvas.create_text(
            250, 150,
            text="Open a GIF image file to begin\n(File → Open  or  Ctrl+O)",
            fill="#888888", font=("Helvetica", 14), justify="center",
            tags="placeholder",
        )

        self._canvas.bind("<Configure>", self._on_canvas_resize)

    def _build_statusbar(self):
        bar = ttk.Frame(self, relief="sunken")
        bar.pack(side="bottom", fill="x")

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self._status_var, anchor="w",
                  padding=(4, 2)).pack(side="left")

        self._size_var = tk.StringVar()
        ttk.Label(bar, textvariable=self._size_var, anchor="e",
                  padding=(4, 2)).pack(side="right")

    # ------------------------------------------------------------------ open

    def _open(self):
        initial = (os.path.dirname(self._current_path)
                   if self._current_path else os.path.expanduser("~"))

        path = filedialog.askopenfilename(
            parent=self,
            title="Open GIF Image",
            initialdir=initial,
            filetypes=[("GIF Images", "*.gif *.GIF"), ("All Files", "*.*")],
        )
        if not path:
            return
        self._load_image(path, add_to_history=True)

    def _load_image(self, path: str, *, add_to_history: bool = True):
        try:
            img = tk.PhotoImage(file=path)
        except tk.TclError as exc:
            messagebox.showerror("Image Error",
                                 f"Could not load image:\n{exc}", parent=self)
            return

        self._current_image = img
        self._current_path  = path

        if add_to_history:
            # Truncate forward history then append
            self._history = self._history[: self._history_index + 1]
            self._history.append(path)
            self._history_index = len(self._history) - 1

        self._display_image()
        self._update_nav_buttons()

        filename = os.path.basename(path)
        self._path_var.set(path)
        self.title(f"Image Browser — {filename}")
        self._status_var.set(f"Loaded: {filename}")
        self._size_var.set(f"{img.width()} × {img.height()} px")

    def _display_image(self):
        img = self._current_image
        if img is None:
            return

        self._canvas.delete("all")
        # Centre the image on the canvas
        cx = max(img.width(),  self._canvas.winfo_width())  // 2
        cy = max(img.height(), self._canvas.winfo_height()) // 2
        self._canvas.create_image(cx, cy, anchor="center", image=img,
                                  tags="image")
        self._canvas.configure(
            scrollregion=(0, 0, img.width(), img.height())
        )

    # ------------------------------------------------------------------ view

    def _fit_to_window(self):
        """Zoom the image (via subsample/zoom) to fit the canvas."""
        if self._current_image is None or self._current_path is None:
            return
        # Reload fresh copy and scale it
        img = tk.PhotoImage(file=self._current_path)
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        iw, ih = img.width(), img.height()

        if iw == 0 or ih == 0:
            return

        factor = min(cw / iw, ch / ih)
        if factor < 1:
            # subsample (shrink): factor e.g. 0.5 → every 2nd pixel
            divisor = max(1, round(1 / factor))
            img = img.subsample(divisor, divisor)
        elif factor > 1:
            # zoom (enlarge): integer multiples only
            multiplier = max(1, int(factor))
            img = img.zoom(multiplier, multiplier)

        self._current_image = img
        self._display_image()
        self._size_var.set(f"{img.width()} × {img.height()} px  (fitted)")

    def _actual_size(self):
        """Reload the image at its original resolution."""
        if self._current_path:
            img = tk.PhotoImage(file=self._current_path)
            self._current_image = img
            self._display_image()
            self._size_var.set(f"{img.width()} × {img.height()} px")

    # ---------------------------------------------------------------- history

    def _go_back(self):
        if self._history_index > 0:
            self._history_index -= 1
            self._load_image(self._history[self._history_index],
                             add_to_history=False)

    def _go_forward(self):
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._load_image(self._history[self._history_index],
                             add_to_history=False)

    def _update_nav_buttons(self):
        self._btn_back.config(
            state="normal" if self._history_index > 0 else "disabled"
        )
        self._btn_fwd.config(
            state="normal" if self._history_index < len(self._history) - 1
            else "disabled"
        )

    # --------------------------------------------------------------- events

    def _on_canvas_resize(self, _event):
        # Re-centre the image when the window is resized
        if self._current_image is not None:
            self._display_image()


if __name__ == "__main__":
    app = ImageBrowser()
    app.mainloop()
