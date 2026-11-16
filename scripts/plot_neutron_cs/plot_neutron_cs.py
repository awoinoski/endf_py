import numpy as np
import matplotlib.pyplot as plt
import endf
import sys
from pathlib import Path
from dataclasses import dataclass

home_dir = Path(__file__).resolve().parent
su_dir = home_dir / "setup_files"

ALLOWED_KEYS = {"isotope", "reaction", "output", "cs_floor"}

REACTION_LABELS = {
    1:   "total",
    2:   "elastic",
    4:   "inelastic",
    16:  "(n,2n)",
    28:  "(n,np)",
    102: "(n,γ)",
    103: "(n,p)",
    107: "(n,α)",
}

DEFAULT_REACTIONS = list(REACTION_LABELS)
CS_FLOOR_DEFAULT = 1e-6

N_CS_DIR = home_dir.parent.parent / "endf_data" / "neutrons-version.VIII.1"


@dataclass(frozen=True)
class Isotope:
    Z: int
    symbol: str
    A: int

    def filename(self):
        return f"n-{self.Z:03d}_{self.symbol}_{self.A:03d}.endf"


@dataclass
class CrossSection:
    mt: int
    label: str
    energy: np.ndarray
    cs: np.ndarray


#--------------------------------------------------------------------------------------#
# Parse setup.par file
def get_setup_file(filename):
    params = {}

    with filename.open("r") as f:
        for line_num, line in enumerate(f, start=1):

            line = line.split("#")[0].strip()
            if not line:
                continue

            tokens = line.split()
            key = tokens[0].lower()
            values = tokens[1:]

            if key not in ALLOWED_KEYS:
                print(f"Line {line_num}: unknown key '{tokens[0]}'", file=sys.stderr)
                sys.exit(1)

            if key == "isotope":
                if len(values) != 3:
                    print(f"Line {line_num}: expected 'isotope <Z> <symbol> <A>'",
                          file=sys.stderr)
                    sys.exit(1)

                try:
                    Z = int(values[0])
                except ValueError:
                    print(f"Line {line_num}: Z '{values[0]}' is not an integer",
                          file=sys.stderr)
                    sys.exit(1)

                symbol = values[1].capitalize()

                try:
                    A = int(values[2])
                except ValueError:
                    print(f"Line {line_num}: mass number '{values[2]}' is not an integer",
                          file=sys.stderr)
                    sys.exit(1)

                if not (1 <= Z <= 118) or A < Z:
                    print(f"Line {line_num}: unphysical values Z={Z}, A={A}",
                          file=sys.stderr)
                    sys.exit(1)

                iso = Isotope(Z, symbol, A)
                isotopes = params.setdefault("isotope", [])

                if iso in isotopes:
                    print(f"Line {line_num}: {symbol}-{A} listed twice, ignoring duplicate",
                          file=sys.stderr)
                else:
                    isotopes.append(iso)

            elif key == "reaction":
                if not values:
                    print(f"Line {line_num}: expected 'reaction <MT> [<MT> ...]'",
                          file=sys.stderr)
                    sys.exit(1)

                reactions = params.setdefault("reaction", [])
                for tok in values:
                    try:
                        mt = int(tok)
                    except ValueError:
                        print(f"Line {line_num}: MT '{tok}' is not an integer",
                              file=sys.stderr)
                        sys.exit(1)

                    if mt not in REACTION_LABELS:
                        print(f"Line {line_num}: MT {mt} is not in the reaction table "
                              f"{sorted(REACTION_LABELS)}", file=sys.stderr)
                        sys.exit(1)

                    if mt not in reactions:
                        reactions.append(mt)

            elif key == "cs_floor":
                if len(values) != 1:
                    print(f"Line {line_num}: expected 'cs_floor <barns>'",
                          file=sys.stderr)
                    sys.exit(1)

                try:
                    floor = float(values[0])
                except ValueError:
                    print(f"Line {line_num}: cs_floor '{values[0]}' is not a number",
                          file=sys.stderr)
                    sys.exit(1)

                if floor <= 0:
                    print(f"Line {line_num}: cs_floor must be positive (log axes)",
                          file=sys.stderr)
                    sys.exit(1)

                params["cs_floor"] = floor

            elif key == "output":
                if len(values) != 1:
                    print(f"Line {line_num}: expected 'output <filename>'",
                          file=sys.stderr)
                    sys.exit(1)

                params["output"] = values[0]

    if "isotope" not in params:
        print("Setup file lists no isotopes!", file=sys.stderr)
        sys.exit(1)

    params.setdefault("reaction", list(DEFAULT_REACTIONS))
    params.setdefault("cs_floor", CS_FLOOR_DEFAULT)
    params.setdefault("output", None)

    return params


#--------------------------------------------------------------------------------------#
# Check that data files exist
def find_data(isotopes, data_dir=N_CS_DIR):
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    found = {}
    missing = []

    for iso in isotopes:
        path = data_dir / iso.filename()
        if path.is_file():
            found[iso] = path
        else:
            missing.append(iso)

    for iso in missing:
        print(f"Missing data for {iso.symbol}-{iso.A}: "
              f"no file {iso.filename()} in {data_dir}", file=sys.stderr)

    return found, missing


#--------------------------------------------------------------------------------------#
# Load cross-section data
def load_material(path):
    try:
        return endf.Material(str(path))
    except Exception as err:
        print(f"Could not read {path}: {err}", file=sys.stderr)
        return None


def get_cross_section(mat, mt):
    if (3, mt) not in mat:
        return None
    tab = mat[3, mt]["sigma"]
    return CrossSection(mt, REACTION_LABELS[mt], np.asarray(tab.x), np.asarray(tab.y))


def load_all(found, reactions):
    data = {}

    for iso, path in found.items():
        mat = load_material(path)
        if mat is None:
            continue

        channels = {}
        for mt in reactions:
            chan = get_cross_section(mat, mt)
            if chan is None:
                print(f"{iso.symbol}-{iso.A}: MT {mt} ({REACTION_LABELS[mt]}) "
                      f"not in evaluation", file=sys.stderr)
                continue
            channels[mt] = chan

        data[iso] = channels

    return data


#--------------------------------------------------------------------------------------#
# Plotting
def log_safe(chan, floor):
    keep = (chan.energy > 0) & (chan.cs > floor)   # log axes can't show zeros
    return chan.energy[keep], chan.cs[keep]


def style_axes(ax, title):
    ax.set_xlabel("Neutron energy (eV)")
    ax.set_ylabel("Cross section (barns)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()


def plot_by_isotope(data, floor):
    figs = []

    for iso, channels in data.items():
        if not channels:
            print(f"{iso.symbol}-{iso.A}: no data to plot", file=sys.stderr)
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        for chan in channels.values():
            e, cs = log_safe(chan, floor)
            if e.size:
                ax.loglog(e, cs, label=chan.label, lw=1)

        style_axes(ax, f"{iso.symbol}-{iso.A} neutron cross sections (ENDF)")
        figs.append((f"{iso.symbol}{iso.A}", fig))

    return figs


def save_or_show(figs, output):
    if not figs:
        print("Nothing to plot!", file=sys.stderr)
        sys.exit(1)

    if output is None:
        plt.show()
        return

    out = home_dir /"figures"/output
    for name, fig in figs:
        path = out.with_name(f"{out.stem}_{name}{out.suffix}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)


#--------------------------------------------------------------------------------------#
def main():

    if len(sys.argv) != 2:
        print("Wrong number of arguments!\n", file=sys.stderr)
        print("python3 plot_neutron_cs.py setup_filename", file=sys.stderr)
        sys.exit(1)

    su_f = su_dir / sys.argv[1]

    if not su_f.is_file():
        print("Setup file not found! Exiting!", file=sys.stderr)
        sys.exit(1)

    params = get_setup_file(su_f)
    found, missing = find_data(params["isotope"])

    if missing:
        print(f"{len(missing)} of {len(params['isotope'])} isotopes missing. Exiting!",
              file=sys.stderr)
        sys.exit(1)

    data = load_all(found, params["reaction"])
    figs = plot_by_isotope(data, params["cs_floor"])
    save_or_show(figs, params["output"])


#--------------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
