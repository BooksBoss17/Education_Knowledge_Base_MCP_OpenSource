"""Version-scoped workaround for accelerated PyMuPDF text-trace ownership.

PyMuPDF 1.27.2.3's extra device consumes references to None for each span
(linewidth is None). Repeated book-page extraction can crash the interpreter.
Its Python device uses the same MuPDF paint operations and returns the same
fields used by our consumers; linewidth is numeric instead of None.
"""


def safe_text_trace(page):
    """Return unrotated trace, restoring page rotation even on device failure."""
    import pymupdf

    if pymupdf.VersionBind != "1.27.2.3" or not isinstance(page, pymupdf.Page):
        return page.get_texttrace()

    pymupdf.CheckParent(page)
    rotation = page.rotation
    try:
        if rotation:
            page.set_rotation(0)
        result = []
        mupdf = pymupdf.mupdf
        class TextTraceDevice(pymupdf.JM_new_texttrace_device):
            # This release's Python callback predates the added context arg.
            def ignore_text(self, context, text, matrix):
                pymupdf.jm_lineart_ignore_text(self, text, matrix)

        device = TextTraceDevice(result)
        try:
            bounds = mupdf.fz_bound_page(page.this)
            device.ptm = mupdf.FzMatrix(1, 0, 0, -1, 0, bounds.y1)
            mupdf.fz_run_page(page.this, device, mupdf.FzMatrix(), mupdf.FzCookie())
        finally:
            mupdf.fz_close_device(device)
        return result
    finally:
        if rotation:
            page.set_rotation(rotation)
