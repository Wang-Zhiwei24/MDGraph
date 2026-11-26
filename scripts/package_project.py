"""Package the release using an explicit list of source and example directories."""

from pathlib import Path
from runpy import run_path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT_FILES = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirement.txt",
    "requirements-tested.txt",
)
SOURCE_DIRS = ("src", "scripts", "tests", "docs", "examples")
EXCLUDED_SUFFIXES = {".key", ".lammpstrj", ".log", ".pem", ".pyc", ".restart", ".sqlite"}


def release_files(root: Path):
    for filename in ROOT_FILES:
        yield root / filename
    for directory in SOURCE_DIRS:
        for path in sorted((root / directory).rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or path.suffix in EXCLUDED_SUFFIXES:
                continue
            if any(
                block.startswith(".") or block == "__pycache__" or block.endswith(".egg-info")
                for block in relative.parts
            ):
                continue
            yield path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = run_path(str(root / "src/mdgraph/__init__.py"))["__version__"]
    release = f"MDGraph-{version}"
    archive = root / f"{release}-source.zip"
    files = list(release_files(root))
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        for path in files:
            output.write(path, f"{release}/{path.relative_to(root).as_posix()}")
    print(f"{archive}\n{len(files)} source, example and documentation files")


if __name__ == "__main__":
    main()
