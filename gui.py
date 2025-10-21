"""
Cleaned Tkinter GUI for the 1D Spring FEA solver.
- Left column: two stacked panels (Elements, Forces/BCs), each taking exactly half of the left column height and both scrollable.
- Right column: Sketch on top, Results (Text/Tables) on bottom.
- No resizable sash; uses grid weights so the stacked panels always fill.

Requires: numpy, Pillow, tkinter
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont
import numpy as np
import os, csv
from PIL import Image, ImageDraw

# ---- import your core FEA pieces ----
# These must exist in your project
from fea_core import Node, SpringElement, SpringFEASolver


# ------------------------------------------------------------
# Scrollable frame helper (vertical only)
# ------------------------------------------------------------
class ScrollFrame(ttk.Frame):
    def __init__(self, master, *, height: int | None = None):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas)

        # Track content size
        self.body.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)

        # Layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        if height is not None:
            self.canvas.configure(height=height)

        # Resize inner frame width to match canvas width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Wheel support when mouse is over the body
        self.body.bind("<Enter>", self._bind_wheel)
        self.body.bind("<Leave>", self._unbind_wheel)

        self.after_idle(lambda: self.canvas.yview_moveto(0.0))


    def _on_canvas_configure(self, event):
        try:
            self.canvas.itemconfigure(self._win, width=event.width)
        except Exception:
            pass

    def _bind_wheel(self, *_):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)  # Linux
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_wheel(self, *_):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        # Windows / Mac send delta, Linux often uses Button-4/5
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.canvas.yview_scroll(-1, "units")
        else:
            self.canvas.yview_scroll(1, "units")

    def _recompute_scrollregion(self, *_):
        content_w = max(self.body.winfo_reqwidth(), 1)
        content_h = max(self.body.winfo_reqheight(), 1)
        self.canvas.configure(scrollregion=(0, 0, content_w, content_h))

        canvas_h = max(self.canvas.winfo_height(), 1)
        self._overflow = content_h > canvas_h + 1

        # enable/disable scrollbar
        self.vsb.configure(state=("normal" if self._overflow else "disabled"))

        # if no overflow OR content just shrank below viewport, snap to top
        if not self._overflow:
            self.canvas.yview_moveto(0.0)
        else:
            # if we were at an invalid fraction (sometimes happens when content shrinks),
            # clamp to [0, 1] and prefer top on shrink
            first, last = self.canvas.yview()
            if first > 0.999 or first < 0.0 or last - first >= 1.0:
                self.canvas.yview_moveto(0.0)

    def scroll_to_top(self):
        self.canvas.yview_moveto(0.0)


class ToolTip:
    """Simple tooltip that appears when hovering over a widget."""
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay  # milliseconds before showing
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._unschedule)
        widget.bind("<ButtonPress>", self._unschedule)

    def _schedule(self, event=None):
        self._unschedule()
        self.id = self.widget.after(self.delay, self._show)

    def _unschedule(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        self._hide()

    def _show(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # no window frame
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9)
        )
        label.pack(ipadx=6, ipady=3)

    def _hide(self):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

# ------------------------------------------------------------
# The App
# ------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("1D Spring FEA Solver — Clean UI")
        self.geometry("1000x700")
        self.minsize(900, 650)

        # Theme
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        # State
        self.num_nodes_var = tk.IntVar(value=3)
        self.num_elems_var = tk.IntVar(value=2)
        self.view_mode_var = tk.StringVar(value="undeformed")  # 'undeformed' or 'deformed'
        self.auto_solve_var = tk.BooleanVar(value=True)

        # Build UI
        self._build_topbar()
        self._build_columns()

        # Initialize tables
        self.rebuild_element_table()
        self.rebuild_force_bc_table()

    # ---------------- topbar ----------------
    def _build_topbar(self):
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=12, pady=8)

        ttk.Label(bar, text="Nodes:").pack(side=tk.LEFT)
        ttk.Entry(bar, width=6, textvariable=self.num_nodes_var).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(bar, text="Elements:").pack(side=tk.LEFT)
        ttk.Entry(bar, width=6, textvariable=self.num_elems_var).pack(side=tk.LEFT, padx=(4, 16))

        btn_apply = ttk.Button(bar, text="Apply Counts", command=self.apply_counts)
        btn_apply.pack(side=tk.LEFT)
        ToolTip(btn_apply, "Rebuild the tables for the number of nodes/elements entered.")

        btn_solve = ttk.Button(bar, text="Solve", command=self.solve_model)
        btn_solve.pack(side=tk.LEFT, padx=(16, 0))
        ToolTip(btn_solve, "Run the solver to compute displacements, reactions, and element forces.")

        btn_export = ttk.Button(bar, text="Export CSVs", command=self.export_results_csv)
        btn_export.pack(side=tk.LEFT, padx=8)
        ToolTip(btn_export, "Save node, element, and stiffness data to CSV files.")

        chk_auto = ttk.Checkbutton(bar, text="Auto-solve on edit", variable=self.auto_solve_var)
        chk_auto.pack(side=tk.RIGHT)
        ToolTip(chk_auto, "Automatically re-solve the model whenever inputs change.")

    # ---------------- columns layout ----------------
    def _build_columns(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        # Left and right columns
        left = ttk.Frame(container)
        right = ttk.Frame(container)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        ttk.Separator(container, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Fix left width but allow vertical growth
        LEFT_WIDTH = 520
        left.configure(width=LEFT_WIDTH)
        left.pack_propagate(False)

        # LEFT: two stacked panels, each exactly half height
        left.grid_rowconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        lf_e = ttk.LabelFrame(left, text="Elements (i, j, Stiffness k)")
        lf_f = ttk.LabelFrame(left, text="Nodal Forces & Boundary Conditions")
        lf_e.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
        lf_f.grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))

        # Each half contains a scrollable region
        self.elem_sf = ScrollFrame(lf_e)
        self.elem_sf.pack(fill=tk.BOTH, expand=True)
        self.elem_table = self.elem_sf.body

        self.force_sf = ScrollFrame(lf_f)
        self.force_sf.pack(fill=tk.BOTH, expand=True)
        self.force_table = self.force_sf.body

        # Optional hint line at the bottom of BC panel (inside label frame)
        self.bc_hint_var = tk.StringVar(value="")
        self.bc_hint_lbl = ttk.Label(lf_f, textvariable=self.bc_hint_var, foreground="#b45309", justify="left")
        self.bc_hint_lbl.pack(fill=tk.X, padx=12, pady=(4, 6), anchor="w")
        lf_f.bind(
            "<Configure>",
            lambda e: self.bc_hint_lbl.configure(wraplength=max(200, e.width - 24)),
        )

        # RIGHT: sketch on top, results bottom
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=4)
        right.grid_columnconfigure(0, weight=1)

        rf_sketch = ttk.LabelFrame(right, text="Sketch")
        rf_sketch.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))

        viewbar = ttk.Frame(rf_sketch)
        viewbar.pack(side=tk.TOP, anchor="w", padx=8, pady=(6, 0))
        ttk.Radiobutton(viewbar, text="Undeformed", value="undeformed", variable=self.view_mode_var, command=self.on_view_change).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(viewbar, text="Deformed", value="deformed", variable=self.view_mode_var, command=self.on_view_change).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(rf_sketch, bg="#ffffff", height=320)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        rf_res = ttk.LabelFrame(right, text="Results")
        rf_res.grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))

        self.results_nb = ttk.Notebook(rf_res)
        self.results_nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Text tab
        self.tab_text = ttk.Frame(self.results_nb)
        self.results_nb.add(self.tab_text, text="Text")
        self.results_text = tk.Text(self.tab_text, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.configure(state=tk.DISABLED)

        # Tables tab
        self.tab_tables = ttk.Frame(self.results_nb)
        self.results_nb.add(self.tab_tables, text="Tables")
        self.tables_nb = ttk.Notebook(self.tab_tables)
        self.tables_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Nodal table
        self.tab_nodes = ttk.Frame(self.tables_nb)
        self.tables_nb.add(self.tab_nodes, text="Nodal Results")
        nodes_wrap = ttk.Frame(self.tab_nodes)
        nodes_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.tv_nodes = ttk.Treeview(nodes_wrap, columns=("node","u","R","fixed","F"), show="headings", height=8)
        for k, w in zip(["node","u","R","fixed","F"], [60,120,120,70,120]):
            self.tv_nodes.heading(k, text={"node":"Node","u":"u","R":"R","fixed":"Fixed?","F":"F"}[k])
            self.tv_nodes.column(k, width=w, anchor=tk.CENTER)
        sb_nodes = ttk.Scrollbar(nodes_wrap, orient="vertical", command=self.tv_nodes.yview)
        self.tv_nodes.configure(yscrollcommand=sb_nodes.set)
        self.tv_nodes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_nodes.pack(side=tk.RIGHT, fill=tk.Y)
        self._setup_treeview_striping(self.tv_nodes)

        # Element table
        self.tab_elems = ttk.Frame(self.tables_nb)
        self.tables_nb.add(self.tab_elems, text="Element Results")
        elems_wrap = ttk.Frame(self.tab_elems)
        elems_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.tv_elems = ttk.Treeview(elems_wrap, columns=("e","i","j","k","axial","Fi","Fj"), show="headings", height=10)
        heads = {"e":"e#","i":"i","j":"j","k":"Stiffness (k)","axial":"Axial","Fi":"F_i","Fj":"F_j"}
        widths = {"e":60,"i":60,"j":60,"k":100,"axial":120,"Fi":120,"Fj":120}
        for col in ("e","i","j","k","axial","Fi","Fj"):
            self.tv_elems.heading(col, text=heads[col])
            self.tv_elems.column(col, width=widths[col], anchor=tk.CENTER)
        sb_elems = ttk.Scrollbar(elems_wrap, orient="vertical", command=self.tv_elems.yview)
        self.tv_elems.configure(yscrollcommand=sb_elems.set)
        self.tv_elems.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_elems.pack(side=tk.RIGHT, fill=tk.Y)
        self._setup_treeview_striping(self.tv_elems)

        # Global K
        self.tab_K = ttk.Frame(self.tables_nb)
        self.tables_nb.add(self.tab_K, text="Global K")
        self.k_frame = ttk.Frame(self.tab_K)
        self.k_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.tv_K = None

    # ---------------- dynamic tables ----------------
    def _clear_children(self, container: ttk.Frame):
        for w in container.winfo_children():
            w.destroy()

    def _snapshot_elements(self):
        data = []
        for trip in getattr(self, "elem_entries", []):
            try:
                i = int(trip[0].get()); j = int(trip[1].get()); k = float(trip[2].get())
                data.append((i, j, k))
            except Exception:
                data.append(None)
        return data

    def _snapshot_nodes(self):
        forces, bc_types, uvals = [], [], []
        for idx in range(len(getattr(self, "force_vars", []))):
            try:
                forces.append(float(self.force_vars[idx].get()))
            except Exception:
                forces.append(0.0)
            try:
                bc_types.append(str(self.bc_type_vars[idx].get()))
            except Exception:
                bc_types.append("Free")
            try:
                uvals.append(float(self.u_val_vars[idx].get()))
            except Exception:
                uvals.append(0.0)
        return forces, bc_types, uvals

    def rebuild_element_table(self):
        old = self._snapshot_elements()
        self._clear_children(self.elem_table)
        nE = max(0, self.num_elems_var.get())

        headers = ["#", "i", "j", "Stiffness (k)"]
        for c, h in enumerate(headers):
            ttk.Label(self.elem_table, text=h, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=c, padx=4, pady=4)

        self._suspend_traces = True
        self.elem_entries = []
        for r in range(1, nE + 1):
            ttk.Label(self.elem_table, text=str(r)).grid(row=r, column=0, padx=2, pady=2)
            i_def, j_def, k_def = r, r + 1, 1.0
            if r - 1 < len(old) and old[r - 1] is not None:
                i_def, j_def, k_def = old[r - 1]
            i_var = tk.IntVar(value=i_def)
            j_var = tk.IntVar(value=j_def)
            k_var = tk.DoubleVar(value=k_def)
            ttk.Entry(self.elem_table, width=6, textvariable=i_var).grid(row=r, column=1, padx=2, pady=2)
            ttk.Entry(self.elem_table, width=6, textvariable=j_var).grid(row=r, column=2, padx=2, pady=2)
            ttk.Entry(self.elem_table, width=10, textvariable=k_var).grid(row=r, column=3, padx=2, pady=2)
            i_var.trace_add("write", self._on_var_change)
            j_var.trace_add("write", self._on_var_change)
            k_var.trace_add("write", self._on_var_change)
            self.elem_entries.append((i_var, j_var, k_var))
        self._suspend_traces = False
        self.elem_sf.scroll_to_top()
        self._schedule_refresh()

    def rebuild_force_bc_table(self):
        try:
            oldF, oldBC, oldU = self._snapshot_nodes()
        except Exception:
            oldF, oldBC, oldU = [], [], []
        self._clear_children(self.force_table)
        nN = max(1, self.num_nodes_var.get())

        headers = ["Node", "Load F", "BC type", "u value"]
        for c, h in enumerate(headers):
            ttk.Label(self.force_table, text=h, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=c, padx=4, pady=4)

        self._suspend_traces = True
        self.force_vars, self.bc_type_vars, self.u_val_vars = [], [], []
        for r in range(1, nN + 1):
            ttk.Label(self.force_table, text=str(r)).grid(row=r, column=0, padx=2, pady=2)
            f_def = oldF[r - 1] if r - 1 < len(oldF) else 0.0
            bc_def = oldBC[r - 1] if r - 1 < len(oldBC) else ("Fixed" if r == 1 else "Free")
            u_def = oldU[r - 1] if r - 1 < len(oldU) else 0.0

            f_var = tk.DoubleVar(value=f_def)
            entF = ttk.Entry(self.force_table, width=12, textvariable=f_var)
            entF.grid(row=r, column=1, padx=2, pady=2)
            self.force_vars.append(f_var)
            f_var.trace_add("write", self._on_var_change)

            bc_var = tk.StringVar(value=bc_def)
            cbx = ttk.Combobox(
                self.force_table,
                values=["Free", "Fixed", "Prescribed"],
                state="readonly",
                width=12,
                textvariable=bc_var,
            )
            cbx.grid(row=r, column=2, padx=2, pady=2)
            self.bc_type_vars.append(bc_var)

            u_var = tk.DoubleVar(value=u_def)
            entU = ttk.Entry(self.force_table, width=10, textvariable=u_var)
            entU.grid(row=r, column=3, padx=2, pady=2)
            self.u_val_vars.append(u_var)

            def _toggle(*_, entry=entU, var_bc=bc_var):
                entry.configure(state=("normal" if var_bc.get() == "Prescribed" else "disabled"))
                self._on_var_change()

            _toggle()
            bc_var.trace_add("write", _toggle)
            u_var.trace_add("write", self._on_var_change)
        self._suspend_traces = False
        self.force_sf.scroll_to_top()
        self._schedule_refresh()

    # ---------------- actions ----------------
    def _update_bc_hint(self):
        """Warn when a node has a non-zero Load F together with a constrained displacement."""
        if not hasattr(self, "bc_hint_var"):
            return
        try:
            eps = 1e-12
            hints = []
            nN = len(getattr(self, "force_vars", []))
            for r in range(nN):
                # force value
                f = 0.0
                try:
                    f = float(self.force_vars[r].get())
                except Exception:
                    pass
                # bc choice
                bc = self.bc_type_vars[r].get() if r < len(getattr(self, "bc_type_vars", [])) else "Free"
                if bc in ("Fixed", "Prescribed") and abs(f) > eps:
                    hints.append(
                        f"Node {r+1}: Load F with {bc} u — treated as reaction only (element forces unchanged)."
                    )
            self.bc_hint_var.set("⚠️  " + "  ".join(hints) if hints else "")
        except Exception:
            self.bc_hint_var.set("")

    def on_view_change(self):
        self.refresh_model()

    def _on_var_change(self, *args):
        if getattr(self, "_suspend_traces", False):
            return
        try:
            self._update_bc_hint()
        except Exception:
            pass
        self._schedule_refresh()

    def _schedule_refresh(self):
        try:
            if hasattr(self, "_refresh_after_id") and self._refresh_after_id:
                self.after_cancel(self._refresh_after_id)
        except Exception:
            pass
        try:
            self._update_bc_hint()
        except Exception:
            pass
        self._refresh_after_id = self.after(150, self.refresh_model)

    def refresh_model(self):
        try:
            nodes, elements = self.collect_model()
        except Exception:
            self.draw_sketch()
            return
        fixed_flags = [nd.fixed for nd in nodes]
        if self.auto_solve_var.get():
            if self.view_mode_var.get() == "deformed":
                self.solve_model()
            else:
                self.draw_sketch(u=None, fixed=fixed_flags)
        else:
            if self.view_mode_var.get() == "deformed" and hasattr(self, "_last_u"):
                self.draw_sketch(u=self._last_u, fixed=fixed_flags)
            else:
                self.draw_sketch(u=None, fixed=fixed_flags)

    def apply_counts(self):
        try:
            nN = int(self.num_nodes_var.get()); nE = int(self.num_elems_var.get())
            if nN < 1 or nE < 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid", "Please enter valid positive integers for nodes/elements.")
            return
        self.rebuild_element_table()
        self.rebuild_force_bc_table()
        self.draw_sketch()

    def collect_model(self):
        nN = int(self.num_nodes_var.get())
        nodes = [Node(i + 1) for i in range(nN)]
        # BCs & loads
        for r in range(nN):
            try:
                nodes[r].force = float(self.force_vars[r].get())
            except Exception:
                raise ValueError("Forces must be numeric.")
            bc = str(self.bc_type_vars[r].get()) if r < len(self.bc_type_vars) else "Free"
            uv = 0.0
            if bc == "Prescribed":
                try:
                    uv = float(self.u_val_vars[r].get())
                except Exception:
                    raise ValueError("Prescribed u values must be numeric.")
            if bc == "Fixed":
                nodes[r].fixed = True; nodes[r].prescribed = False; nodes[r].u_prescribed = 0.0
            elif bc == "Prescribed":
                nodes[r].fixed = True; nodes[r].prescribed = True; nodes[r].u_prescribed = uv
            else:
                nodes[r].fixed = False; nodes[r].prescribed = False; nodes[r].u_prescribed = 0.0

        # Elements
        elements = []
        for (i_var, j_var, k_var) in self.elem_entries:
            try:
                i = int(i_var.get()); j = int(j_var.get()); k = float(k_var.get())
            except Exception:
                raise ValueError("Element data must be numeric (i, j integers; k float).")
            if i < 1 or j < 1 or i > nN or j > nN:
                raise ValueError(f"Element node indices must be between 1 and {nN}. Got ({i}, {j}).")
            elements.append(SpringElement(nodes[i - 1], nodes[j - 1], k))

        if not any(nd.fixed for nd in nodes):
            if not messagebox.askyesno("No supports", "No nodes are fixed. Proceed anyway? (Matrix may be singular)"):
                raise RuntimeError("Aborted by user")
        return nodes, elements

    def solve_model(self):
        try:
            nodes, elements = self.collect_model()
            solver = SpringFEASolver(nodes, elements)
            K = solver.assemble()
            u, R, free_idx, fixed_idx = solver.solve()
            elem_forces = solver.element_forces()
            elem_end_forces = [e.nodal_actions(u) for e in elements]
            self._last_nodes = nodes; self._last_elements = elements; self._last_u = u
        except RuntimeError:
            return
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        # Text results
        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Global stiffness matrix K:\n")
        self.results_text.insert(
            tk.END, np.array2string(K, formatter={"float_kind": lambda x: f"{x:10.4g}"}) + "\n\n"
        )
        self.results_text.insert(tk.END, "Displacements u:\n")
        self.results_text.insert(
            tk.END, np.array2string(u, formatter={"float_kind": lambda x: f"{x:10.6g}"}) + "\n\n"
        )
        self.results_text.insert(tk.END, "Reactions R:\n")
        self.results_text.insert(
            tk.END, np.array2string(R, formatter={"float_kind": lambda x: f"{x:10.6g}"}) + "\n\n"
        )
        self.results_text.insert(tk.END, "Element axial forces (+ = tension):\n")
        ef_str = np.array2string(np.array(elem_forces), formatter={"float_kind": lambda x: f"{x:10.6g}"})
        self.results_text.insert(tk.END, ef_str + "\n\n")
        self.results_text.insert(tk.END, "Element end forces [F_i, F_j]:\n")
        eef_str = np.array2string(np.array(elem_end_forces), formatter={"float_kind": lambda x: f"{x:10.6g}"})
        self.results_text.insert(tk.END, eef_str + "\n\n")
        self.results_text.insert(tk.END, f"Free DOFs: {[int(i)+1 for i in list(free_idx)]}\n")
        self.results_text.insert(tk.END, f"Fixed DOFs: {[int(i)+1 for i in list(fixed_idx)]}\n")

        # Note about loads on constrained DOFs
        try:
            eps = 1e-12
            flagged = []
            for i in range(len(getattr(self, "bc_type_vars", []))):
                bc = self.bc_type_vars[i].get()
                fval = 0.0
                try:
                    fval = float(self.force_vars[i].get())
                except Exception:
                    pass
                if bc in ("Fixed", "Prescribed") and abs(fval) > eps:
                    flagged.append(str(i + 1))
            if flagged:
                self.results_text.insert(
                    tk.END,
                    "Note: Loads at constrained DOFs (nodes "
                    + ", ".join(flagged)
                    + ") are treated as reactions only and do not change element forces.\n",
                )
        except Exception:
            pass
        self.results_text.configure(state=tk.DISABLED)

        # Tables
        self.populate_results_tables(nodes, elements, K, u, R, elem_forces, elem_end_forces)

        # Sketch
        fixed_flags = [nd.fixed for nd in nodes]
        u_vec = np.array([nd.u for nd in nodes])
        self.draw_sketch(u=u_vec if self.view_mode_var.get() == "deformed" else None, fixed=fixed_flags)

    # -------- tables helpers --------
    def _setup_treeview_striping(self, tv):
        try:
            tv.tag_configure("even", background="#ffffff")
            tv.tag_configure("odd", background="#f2f4f8")
        except Exception:
            pass

    def populate_results_tables(self, nodes, elements, K, u, R, elem_forces, elem_end_forces):
        # nodes
        for item in self.tv_nodes.get_children():
            self.tv_nodes.delete(item)
        for i, nd in enumerate(nodes):
            self.tv_nodes.insert(
                "",
                tk.END,
                values=(
                    i + 1,
                    f"{u[i]:.6g}",
                    f"{R[i]:.6g}",
                    "Yes" if nd.fixed else "No",
                    f"{nd.force:.6g}",
                ),
                tags=("even" if i % 2 == 0 else "odd",),
            )
        # elements
        for item in self.tv_elems.get_children():
            self.tv_elems.delete(item)
        for eidx, e in enumerate(elements, start=1):
            Fi, Fj = elem_end_forces[eidx - 1]
            idx0 = eidx - 1
            self.tv_elems.insert(
                "",
                tk.END,
                values=(
                    eidx,
                    e.i.id,
                    e.j.id,
                    f"{e.k:.6g}",
                    f"{elem_forces[idx0]:.6g}",
                    f"{Fi:.6g}",
                    f"{Fj:.6g}",
                ),
                tags=("even" if idx0 % 2 == 0 else "odd",),
            )
        # K matrix
        self._rebuild_K_table(K)

    def _rebuild_K_table(self, K: np.ndarray):
        if self.tv_K is not None:
            try:
                self.tv_K.destroy()
            except Exception:
                pass
        n = K.shape[0]
        cols = [f"c{j+1}" for j in range(n)]
        self.tv_K = ttk.Treeview(self.k_frame, columns=cols, show="headings", height=min(10, n))
        self._setup_treeview_striping(self.tv_K)
        for j in range(n):
            name = cols[j]
            self.tv_K.heading(name, text=str(j + 1))
            self.tv_K.column(name, width=80, anchor=tk.CENTER)
        for i in range(n):
            vals = [f"{K[i, j]:.6g}" for j in range(n)]
            self.tv_K.insert("", tk.END, values=vals, tags=("even" if i % 2 == 0 else "odd",))
        self.tv_K.pack(fill=tk.BOTH, expand=True)

    # ---------------- sketch ----------------
    def draw_sketch(self, u=None, fixed=None):
        accent = "#1f77b4"; line = "#111827"; muted = "#7a7a7a"; text = "#111827"
        self.canvas.update_idletasks()
        self.canvas.delete("all")
        nN = max(1, int(self.num_nodes_var.get()))
        width = self.canvas.winfo_width() - 50 or 600
        height = self.canvas.winfo_height() or 320
        margin_x, margin_y = 40, 40
        usable_w = max(1.0, width - 2 * margin_x)
        usable_h = max(1.0, height - 2 * margin_y)
        y_mid = height / 2.0

        self.FONT_SKETCH_NODE = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.FONT_SKETCH_ELEM = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.FONT_SKETCH_TAG = tkfont.Font(family="Segoe UI", size=12)

        # base x
        if nN == 1:
            xs = [width / 2.0]
        else:
            spacing = usable_w / float(nN - 1)
            xs = [margin_x + i * spacing for i in range(nN)]

        # displacement magnification clamped
        disp_mag = 200.0
        dxs = [0.0] * nN
        if u is not None and len(u) == nN:
            umax = max(abs(float(val)) for val in u) if len(u) > 0 else 0.0
            scale = disp_mag if umax == 0 else disp_mag / (umax if umax != 0 else 1)
            dxs = [float(val) * scale for val in u]
            span = (max(xs[i] + dxs[i] for i in range(nN)) - min(xs[i] + dxs[i] for i in range(nN)))
            if span > usable_w and span > 0:
                f = usable_w / span
                dxs = [d * f for d in dxs]

        # read BC types, forces, and prescribed u (for labels/arrows)
        bc_types, forces, uvals = [], [], []
        for i in range(nN):
            bc_types.append(self.bc_type_vars[i].get())
            try:
                forces.append(float(self.force_vars[i].get()))
            except:
                forces.append(0.0)
            try:
                uvals.append(float(self.u_val_vars[i].get()))
            except:
                uvals.append(0.0)

        # helper: convert index to Unicode subscript (e.g., 2 -> ₂)
        _sub_map = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")
        def sub(n: int) -> str:
            return str(n).translate(_sub_map)

        # helper: draw a small horizontal arrow above a node
        def draw_arrow(x, sign, y=None):
            L = 26  # arrow length
            yy = (y_mid - 28) if y is None else y
            if sign >= 0:
                self.canvas.create_line(x, yy, x + L, yy, arrow=tk.LAST, width=2, fill=text)
                return x + L + 8, yy  # label anchor to the right
            else:
                self.canvas.create_line(x, yy, x - L, yy, arrow=tk.LAST, width=2, fill=text)
                return x - L - 8, yy  # label anchor to the left

        # elements list
        elems = []
        try:
            for (i_var, j_var, _k_var) in getattr(self, "elem_entries", []):
                i = int(i_var.get()) - 1; j = int(j_var.get()) - 1
                if i < 0 or j < 0 or i >= nN or j >= nN or i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                elems.append({"i": i, "j": j, "a": a, "b": b})
        except Exception:
            elems = []

        # track assignment: different tracks if share node or overlapping interval
        tracks = []
        for e_idx, e in enumerate(elems):
            used = set()
            for p_idx in range(e_idx):
                p = elems[p_idx]
                if e["i"] in (p["i"], p["j"]) or e["j"] in (p["i"], p["j"]):
                    used.add(tracks[p_idx])
                if (e["a"] < p["b"] and p["a"] < e["b"]) and not (e["a"] == p["b"] or p["a"] == e["b"]):
                    used.add(tracks[p_idx])
            t = 0
            while t in used:
                t += 1
            tracks.append(t)

        def track_to_level(t: int) -> int:
            if t == 0:
                return 0
            return ((t + 1) // 2) * (1 if t % 2 == 1 else -1)

        max_level = max(abs(track_to_level(t)) for t in tracks) if tracks else 0
        guard = 12.0
        H = max(10.0, usable_h / 2.0 - guard)
        if max_level == 0:
            step = 0.0; amp = min(12.0, H * 0.5)
        else:
            step = (H * 0.70) / float(max_level); step = max(8.0, min(30.0, step))
            remaining = H - max_level * step
            if remaining < 6.0:
                step = (H - 6.0) / float(max_level); step = max(6.0, min(30.0, step))
                remaining = H - max_level * step
            amp = max(6.0, min(12.0, remaining))

        # draw elements (with element id labels)
        for idx, e in enumerate(elems):
            level = track_to_level(tracks[idx])
            y = y_mid + level * step
            xi = xs[e["i"]] + dxs[e["i"]]; xj = xs[e["j"]] + dxs[e["j"]]
            self.canvas.create_line(xi, y_mid, xi, y, fill=muted)
            self.canvas.create_line(xj, y_mid, xj, y, fill=muted)
            self._draw_spring(xi, xj, y, amp=amp, color=line)
            # element id near the spring midspan
            xm = (xi + xj) / 2.0
            label_offset = (amp + 10) * (-1 if level >= 0 else 1)  # above for upper, below for lower
            self.canvas.create_text(xm, y + label_offset, text=str(idx + 1), fill=text, font=self.FONT_SKETCH_NODE)

        # draw nodes (with F_i / u_i labels)
        if fixed is None:
            fixed = [False] * nN
        for idx in range(nN):
            x = xs[idx] + dxs[idx]
            self.canvas.create_oval(x - 6, y_mid - 6, x + 6, y_mid + 6, fill=accent, outline="")
            # node index above
            self.canvas.create_text(x, y_mid - 14, text=str(idx + 1), fill=text, font=self.FONT_SKETCH_ELEM)
            if fixed[idx]:
                self._draw_support(x, y_mid + 8, color=muted)
            # load / prescribed label with arrows (Prescribed takes priority)
            if idx < len(bc_types) and bc_types[idx] == "Prescribed":
                sign = 1 if (idx < len(uvals) and uvals[idx] >= 0) else -1
                xl, yl = draw_arrow(x, sign)
                self.canvas.create_text(
                    xl, yl, text=f"u{sub(idx + 1)}", anchor=("w" if sign >= 0 else "e"), fill=text, font=self.FONT_SKETCH_TAG
                )
            elif idx < len(forces) and abs(forces[idx]) > 1e-12:
                sign = 1 if forces[idx] >= 0 else -1
                xl, yl = draw_arrow(x, sign)
                self.canvas.create_text(
                    xl, yl, text=f"F{sub(idx + 1)}", anchor=("w" if sign >= 0 else "e"), fill=text, font=self.FONT_SKETCH_TAG
                )

    def _draw_spring(self, x1, x2, y, amp=10.0, color=None):
        if x1 == x2:
            return
        if x1 > x2:
            x1, x2 = x2, x1
        length = x2 - x1
        coils = max(6, int(length / 30))
        dx = length / (coils + 1)
        pts = [(x1, y)]
        cur = x1 + dx
        up = True
        for _ in range(coils):
            pts.append((cur, y - amp if up else y + amp))
            up = not up
            cur += dx
        pts.append((x2, y))
        for i in range(len(pts) - 1):
            self.canvas.create_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], fill=color or "#111827")

    def _draw_support(self, x, y, color="#555555"):
        size = 8
        self.canvas.create_polygon(x - size, y, x + size, y, x, y + size, fill=color)

    # ---------------- export ----------------
    def export_results_csv(self):
        """Export results tables to CSV files in a chosen folder.
        Writes: nodes.csv, elements.csv, K.csv.
        If no solution exists yet, this will attempt to solve first.
        """
        # Ensure we have fresh results
        try:
            nodes, elements = self.collect_model()
            solver = SpringFEASolver(nodes, elements)
            K = solver.assemble()
            u, R, _free, _fixed = solver.solve()
            elem_forces = solver.element_forces()
            elem_end_forces = [e.nodal_actions(u) for e in elements]
        except Exception as e:
            messagebox.showerror("Export error", f"Cannot export without a valid solution.\n\n{e}")
            return

        folder = filedialog.askdirectory(title="Choose folder to save CSV results")
        if not folder:
            return

        # Nodes CSV
        nodes_path = os.path.join(folder, "nodes.csv")
        try:
            with open(nodes_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Node", "u", "R", "Fixed", "F"])
                for i, nd in enumerate(nodes, start=1):
                    w.writerow([i, f"{u[i-1]:.6g}", f"{R[i-1]:.6g}", ("Yes" if nd.fixed else "No"), f"{nd.force:.6g}"])
        except Exception as e:
            messagebox.showerror("Export error", f"Failed writing {nodes_path}: {e}")
            return

        # Elements CSV
        elems_path = os.path.join(folder, "elements.csv")
        try:
            with open(elems_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["e#", "i", "j", "k", "axial", "F_i", "F_j"])
                for eidx, e in enumerate(elements, start=1):
                    Fi, Fj = elem_end_forces[eidx-1]
                    w.writerow([eidx, e.i.id, e.j.id, f"{e.k:.6g}", f"{elem_forces[eidx-1]:.6g}", f"{Fi:.6g}", f"{Fj:.6g}"])
        except Exception as e:
            messagebox.showerror("Export error", f"Failed writing {elems_path}: {e}")
            return

        # Global K CSV
        K_path = os.path.join(folder, "K.csv")
        try:
            with open(K_path, "w", newline="") as f:
                w = csv.writer(f)
                # header row: column indices 1..n
                n = K.shape[0]
                w.writerow([" "] + [str(j+1) for j in range(n)])
                for i in range(n):
                    row = [str(i+1)] + [f"{K[i, j]:.6g}" for j in range(n)]
                    w.writerow(row)
        except Exception as e:
            messagebox.showerror("Export error", f"Failed writing {K_path}: {e}")
            return

        messagebox.showinfo("Exported", f"Saved:\n- {nodes_path}\n- {elems_path}\n- {K_path}")


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
