"""Smoke test helpers for itkwidgets rendering.

This script does not attempt to visually render an interactive widget (which
requires a browser front-end). Instead, it verifies that calling
`itkwidgets.view(...)` produces the widget mimebundle that Jupyter uses for
rendering, and writes a small PNG sanity image to confirm the data pipeline is
producing visible output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _version(name: str) -> str:
    """Return an installed package version or '(not installed)'.

    Args:
      name: Distribution name.

    Returns:
      Installed version string.
    """
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "(not installed)"


def _print_versions() -> None:
    """Print key package versions for debugging."""
    versions = {
        "itkwidgets": _version("itkwidgets"),
        "ipywidgets": _version("ipywidgets"),
        "ipydatawidgets": _version("ipydatawidgets"),
        "traitlets": _version("traitlets"),
        "traittypes": _version("traittypes"),
        "notebook": _version("notebook"),
        "jupyter_server": _version("jupyter-server"),
        "numpy": _version("numpy"),
    }
    print("Widget stack versions:")
    for key in sorted(versions):
        print(f"  - {key}=={versions[key]}")


def _write_png(output_path: Path) -> None:
    """Write a small PNG image for sanity checking.

    Args:
      output_path: Path to write the PNG file to.
    """
    # Import lazily so environments without matplotlib can still import this file.
    import matplotlib.pyplot as plt

    # A simple radial gradient with a bright circle in the centre.
    size: int = 192
    y, x = np.ogrid[:size, :size]
    cy: float = (size - 1) / 2.0
    cx: float = (size - 1) / 2.0
    r: np.ndarray = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
    image: np.ndarray = np.clip(1.0 - (r / (size / 2.0)), 0.0, 1.0)

    fig = plt.figure(figsize=(3, 3), dpi=96)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _write_itkwidgets_mimebundle(output_path: Path) -> None:
    """Create a minimal itkwidgets viewer and write its embed state to JSON.

    Args:
      output_path: Path to write the JSON file to.

    Raises:
      RuntimeError: If the viewer does not expose an embed state.
    """
    from itkwidgets import view

    image: np.ndarray = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255
    label: np.ndarray = np.zeros_like(image)
    label[12:20, 12:20] = 1

    try:
        viewer = view(image, label_image=label)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "label_image_weights shape expected" in message:
            raise RuntimeError(
                "itkwidgets failed to construct a Viewer with a label image.\n\n"
                "This usually means your environment is still using an incompatible "
                "widget stack (or an old uv.lock / .venv).\n\n"
                "Recommended fix (run in the examples directory):\n"
                "  1) Delete `.venv/` and `uv.lock`\n"
                "  2) `uv add -r requirements.txt`\n"
                "  3) `uv sync --reinstall`\n\n"
                "Then re-run this smoke test.\n"
            ) from exc
        raise

    # In ipywidgets 7, `_get_embed_state()` is the stable public-ish API we can
    # use to prove the widget has a valid model + state for front-end rendering.
    if not hasattr(viewer, "_get_embed_state"):
        raise RuntimeError("itkwidgets Viewer did not expose `_get_embed_state()`.")

    embed_state: dict[str, Any] = viewer._get_embed_state()  # type: ignore[attr-defined]
    if not isinstance(embed_state, dict) or not embed_state:
        raise RuntimeError("Expected a non-empty embed state from the itkwidgets Viewer.")

    required_keys: tuple[str, ...] = (
        "model_name",
        "model_module",
        "model_module_version",
        "state",
        "buffers",
    )
    missing = [k for k in required_keys if k not in embed_state]
    if missing:
        raise RuntimeError(f"Embed state missing required keys: {missing}")

    output_path.write_text(json.dumps(embed_state, indent=2, sort_keys=True), encoding="utf-8")


def main(*, output_dir: Path) -> int:
    """Run the smoke test and write artefacts.

    Args:
      output_dir: Output directory for artefacts.

    Returns:
      Process exit code (0 for success).
    """
    _print_versions()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_png(output_dir / "widget_smoketest_overlay.png")
    _write_itkwidgets_mimebundle(output_dir / "widget_smoketest_mimebundle.json")
    return 0


if __name__ == "__main__":
    # Write into the repository examples output directory by default.
    default_output_dir = Path(__file__).resolve().parent / "exampleoutput"
    raise SystemExit(main(output_dir=default_output_dir))

