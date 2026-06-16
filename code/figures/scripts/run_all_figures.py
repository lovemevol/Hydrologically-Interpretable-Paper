from __future__ import annotations

from prepare_paperfour_figure_data import main as prepare_main
from paperfour_figure_lib import (
    plot_fig1,
    plot_fig2,
    plot_fig4,
    plot_fig5,
    plot_fig6,
    plot_fig7,
    plot_fig8,
    plot_fig9,
    plot_fig10,
    plot_figS1_parameter_diagnostics,
    plot_figS2_clip_gamma_3d,
    plot_figS3_parallel_robustness,
)


def main() -> None:
    prepare_main()
    for plotter in [
        plot_fig1,
        plot_fig2,
        plot_fig4,
        plot_fig5,
        plot_fig6,
        plot_fig7,
        plot_fig8,
        plot_fig9,
        plot_fig10,
        plot_figS1_parameter_diagnostics,
        plot_figS2_clip_gamma_3d,
        plot_figS3_parallel_robustness,
    ]:
        for path in plotter():
            print(path)


if __name__ == "__main__":
    main()
