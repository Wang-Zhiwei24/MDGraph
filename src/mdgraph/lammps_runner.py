"""Optional portable local smoke-test runner using the LAMMPS Python wheel."""

import argparse
import os


def main():
    from lammps import lammps

    parser = argparse.ArgumentParser()
    parser.add_argument("-in", dest="input", required=True)
    parser.add_argument("-log", default="log.lammps")
    parser.add_argument("-screen", default="screen.log")
    args = parser.parse_args()
    dll_handle = None
    if os.name == "nt" and os.environ.get("MDGRAPH_DLL_DIR"):
        dll_handle = os.add_dll_directory(os.environ["MDGRAPH_DLL_DIR"])
    try:
        instance = lammps(cmdargs=["-log", args.log, "-screen", args.screen])
    except FileNotFoundError as e:
        raise RuntimeError(
            "LAMMPS DLL/dependency missing. On Windows check Microsoft MPI and VC++ runtime; "
            "a project-local verified runtime directory can be set via MDGRAPH_DLL_DIR."
        ) from e
    try:
        instance.file(args.input)
    finally:
        instance.close()
        if dll_handle is not None:
            dll_handle.close()


if __name__ == "__main__":
    main()
