"""Load a CIF, capture a development preview, and exit.

This helper is intentionally outside the application package. It is used for
visual smoke testing while CrystalPath is still launched as a Python program.
"""

from pathlib import Path
import argparse
from time import perf_counter

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from crystalpath.ui.main_window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_cif")
    parser.add_argument("output_directory")
    parser.add_argument("--supercell", nargs=3, type=int, metavar=("A", "B", "C"))
    parser.add_argument("--pick-image", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--camera-check", action="store_true")
    parser.add_argument("--analyze-interstitials", action="store_true")
    parser.add_argument("--interstitial-kind")
    parser.add_argument("--interstitial-site", type=int)
    parser.add_argument("--free-spheres", action="store_true")
    args = parser.parse_args()
    source = Path(args.input_cif).resolve()
    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)

    app = QApplication([])
    window = MainWindow()
    window.show()

    def capture() -> None:
        window.grab().save(str(output / "window.png"))
        window.viewer.screenshot(str(output / "structure.png"))
        window.close()
        app.quit()

    def load() -> None:
        started = perf_counter()
        window.load_cif(str(source))
        load_seconds = perf_counter() - started
        if args.supercell and window.document is not None:
            repeats = tuple(args.supercell)
            window.spin_a.setValue(repeats[0])
            window.spin_b.setValue(repeats[1])
            window.spin_c.setValue(repeats[2])
            window.apply_supercell()
        if args.analyze_interstitials:
            window.show_interstitial_analysis()
            window.run_interstitial_analysis()
            if args.interstitial_kind:
                window.viewer.set_interstitial_kind(args.interstitial_kind)
            if args.interstitial_site and window.interstitial_result is not None:
                selected = next(
                    site
                    for site in window.interstitial_result.sites
                    if site.site_id == args.interstitial_site
                )
                window.viewer.select_interstitial(selected.site_id)
                window._show_interstitial_info(selected)
            if args.free_spheres:
                window.interstitial_spheres_checkbox.setChecked(True)
        if args.pick_image:
            periodic_image = next(
                atom
                for atom in window.viewer._get_scene()[0]
                if atom.offset != (0, 0, 0)
            )
            window.viewer._select_atom(periodic_image)
        if args.benchmark:
            started = perf_counter()
            current = window.viewer.element_radii.get("O", 0.30)
            window.viewer.set_element_radius("O", current)
            redraw_seconds = perf_counter() - started
            print(
                f"load_seconds={load_seconds:.4f} "
                f"redraw_seconds={redraw_seconds:.4f} "
                f"actors={len(window.viewer.renderer.actors)}",
                flush=True,
            )
        if args.camera_check:
            home_angle = window.viewer.camera.GetViewAngle()
            window.viewer.zoom_by(1.20)
            zoomed_angle = window.viewer.camera.GetViewAngle()
            window.viewer.reset_to_home_view()
            reset_angle = window.viewer.camera.GetViewAngle()
            print(
                f"home_angle={home_angle:.4f} "
                f"zoomed_angle={zoomed_angle:.4f} "
                f"reset_angle={reset_angle:.4f}",
                flush=True,
            )
        QTimer.singleShot(3000, capture)

    QTimer.singleShot(500, load)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
