# Modular Launcher

theatRICS starts with a small **Launcher** window rather than immediately
opening the full analysis interface. This keeps startup fast and avoids
loading heavy optional dependencies (such as `tttrlib`, `cellpose`, or
`AFMReader`) until they are actually needed.

---

## The launcher window

The launcher presents three large buttons:

| Button | Opens |
|--------|-------|
| **Correlation Methods** | Image Simulation, RICS Export, RICS Fitting, SFCS, FCS Export, FCS Fitting |
| **AFM** | AFM height profile analysis |
| **Imaging Methods** | ICS, Vesicle Finder, FRAP |

Each button opens a dedicated full-size analysis window containing only
the tabs relevant to that category. All three windows include the
**Results & Logs** tab for logging, session saving, and plot export.

---

## Multiple windows

The launcher window **remains open** regardless of how many category
windows are opened. You can open all three category windows simultaneously
— each is fully independent with its own:

- background worker processes
- progress tracking
- log output
- session state

---

## One instance per category

If you click a button while that category's window is already open, the
existing window is simply **brought to the front** rather than opening a
duplicate.

---

## Closing

Closing the launcher window **closes all open category windows** and
cleans up any running background worker processes before shutting down the
application.

Closing a category window closes only that window and stops its own
background processes, leaving the launcher and other category windows open.

---

## Lazy imports

Each category window only imports the Python modules it actually needs.
For example:

- Opening **AFM** does not import `tttrlib`, `cellpose`, or `pylibCZIrw`.
- Opening **Correlation Methods** does not import `cellpose` or `AFMReader`.
- Opening **Imaging Methods** does not import `tttrlib` or `AFMReader`.

This means the initial startup is fast even if heavy optional packages
(e.g. `numba`, `cellpose`) are installed, since they are only loaded when
the relevant category window is first opened.