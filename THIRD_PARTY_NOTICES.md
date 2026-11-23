# Third-party notices

Project-owned code: AGPL-3.0-only, see LICENSE. Vendored omml2latex parser: Apache-2.0, notice and license in SOURCE/bemarkdown/_vendor/OMML2LATEX_LICENSE.txt and the core wheel. Dependencies are separately installed: PyMuPDF (AGPL/commercial upstream terms), NumPy, Pillow, lxml, BeautifulSoup, psutil, mathml2latex, mathtypejx and transitive dependencies retain their upstream licenses. No GPU runtime binaries or model weights are included. Model and provider licenses/terms apply independently to user-supplied services. Legacy notice copies under licenses are retained for attribution; they do not mean those old assets are bundled in this release.

The API-only runtime installs opencv-python-headless for connected-component image checks. OpenCV uses Apache-2.0 licensing in this release; the Python package includes upstream third-party notices. No OpenCV binaries are bundled in the source archive.

## Optional video dependencies

The Playwright MCP package is installed by npm under its Apache-2.0 license. Qwen model files are obtained from the upstream repositories identified in the model manifests. FFmpeg and aria2 binaries are provisioned by the user under their respective distribution terms. No third-party desktop application is included.
