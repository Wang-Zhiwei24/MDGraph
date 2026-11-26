"""Execute a small, numerical subset of Python on staged trajectory arrays."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.geometry import find_mic
from scipy import signal, spatial

BUILTINS = {
    function.__name__: function
    for function in (
        abs,
        all,
        any,
        bool,
        dict,
        enumerate,
        float,
        int,
        len,
        list,
        max,
        min,
        range,
        reversed,
        round,
        sorted,
        sum,
        tuple,
        zip,
    )
}
LIBRARIES = {
    "np": {
        name: getattr(np, name)
        for name in (
            "abs",
            "arange",
            "argmax",
            "argsort",
            "array",
            "asarray",
            "average",
            "bincount",
            "ceil",
            "clip",
            "concatenate",
            "corrcoef",
            "cos",
            "count_nonzero",
            "cross",
            "diff",
            "digitize",
            "dot",
            "einsum",
            "exp",
            "floor",
            "histogram",
            "isfinite",
            "linspace",
            "log",
            "max",
            "mean",
            "median",
            "min",
            "nanmean",
            "negative",
            "ones",
            "percentile",
            "pi",
            "polyfit",
            "quantile",
            "rint",
            "sin",
            "sqrt",
            "stack",
            "std",
            "sum",
            "unique",
            "var",
            "where",
            "zeros",
        )
    },
    "linalg": {"norm": np.linalg.norm, "det": np.linalg.det, "inv": np.linalg.inv},
    "scipy": {
        "cdist": spatial.distance.cdist,
        "pdist": spatial.distance.pdist,
        "correlate": signal.correlate,
    },
    "ase": {"find_mic": find_mic},
}
METHODS = {
    "astype",
    "reshape",
    "ravel",
    "sum",
    "mean",
    "std",
    "min",
    "max",
    "append",
    "tolist",
    "copy",
    "transpose",
    "all",
    "any",
}
SYNTAX = (
    ast.Assign,
    ast.AugAssign,
    ast.Return,
    ast.If,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.Expr,
    ast.ListComp,
    ast.DictComp,
    ast.comprehension,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.IfExp,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.expr_context,
)


def compile_analysis(script: str):
    """Accept one numerical function and restrict calls to known library operations."""
    tree = ast.parse(script, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Provide only def analyze(data) with no accompanying code")
    function = tree.body[0]
    if ast.dump(function.args) != ast.dump(ast.parse("def analyze(data): pass").body[0].args):
        raise ValueError("Expected exactly one unannotated data argument")
    if function.name != "analyze" or function.decorator_list or function.returns:
        raise ValueError("Use an undecorated function named analyze without annotations")
    for statement in function.body:
        for node in ast.walk(statement):
            if not isinstance(node, SYNTAX):
                raise ValueError(f"Unsupported Python syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and (
                node.id.startswith("__")
                or isinstance(node.ctx, ast.Store)
                and node.id in (LIBRARIES.keys() | BUILTINS.keys() | {"data"})
            ):
                raise ValueError(f"Name is reserved: {node.id}")
            if isinstance(node, ast.Attribute):
                library = LIBRARIES.get(node.value.id) if isinstance(node.value, ast.Name) else None
                if library is not None:
                    if node.attr not in library:
                        raise ValueError(f"Library operation is unavailable: {node.attr}")
                elif node.attr not in METHODS | {"shape", "ndim", "size", "T"}:
                    raise ValueError(f"Attribute is unavailable: {node.attr}")
            if isinstance(node, ast.Call):
                called = node.func
                if not (
                    isinstance(called, ast.Name)
                    and called.id in BUILTINS
                    or isinstance(called, ast.Attribute)
                ):
                    raise ValueError(
                        "Call only supplied numerical operations and basic Python built-ins"
                    )
    return compile(tree, "<generated trajectory analysis>", "exec")


def restrict_linux():
    """Restrict the loaded worker to memory, timing and standard I/O syscalls."""
    import ctypes
    import errno
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (35, 35))
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError("Could not initialize the Linux analysis sandbox")
    try:
        for name in (
            "read",
            "write",
            "close",
            "fstat",
            "newfstatat",
            "lseek",
            "getpid",
            "gettid",
            "futex",
            "sched_yield",
            "clock_gettime",
            "gettimeofday",
            "brk",
            "mmap",
            "mprotect",
            "munmap",
            "mremap",
            "madvise",
            "rt_sigaction",
            "rt_sigprocmask",
            "rt_sigreturn",
            "sigaltstack",
            "exit",
            "exit_group",
        ):
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and seccomp.seccomp_rule_add(context, 0x7FFF0000, number, 0):
                raise RuntimeError(f"Could not allow the {name} sandbox system call")
        if seccomp.seccomp_load(context):
            raise RuntimeError("Could not enable the Linux analysis sandbox")
    finally:
        seccomp.seccomp_release(context)


def run(script_path: Path, data_path: Path) -> dict:
    code = compile_analysis(script_path.read_text(encoding="utf-8"))
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    scope = {
        "__builtins__": BUILTINS,
        **{name: SimpleNamespace(**functions) for name, functions in LIBRARIES.items()},
    }
    if sys.platform == "linux":
        restrict_linux()
    exec(code, scope)
    result = scope["analyze"](data)
    if not isinstance(result, dict) or set(result) != {"columns", "units"}:
        raise ValueError("Return exactly a columns dict and a units dict")
    columns, units = result["columns"], result["units"]
    if not isinstance(columns, dict) or not isinstance(units, dict) or set(columns) != set(units):
        raise ValueError("Column names and units must match")
    if not 1 <= len(columns) <= 16 or not all(
        isinstance(key, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_ .-]{0,63}", key)
        for key in columns
    ):
        raise ValueError("Use 1–16 ASCII column names starting with a letter")
    if not all(
        isinstance(unit, str) and unit.strip() and len(unit) <= 32 for unit in units.values()
    ):
        raise ValueError("Give a short unit for every column, using '1' for dimensionless values")
    arrays = {key: np.asarray(value, dtype=float) for key, value in columns.items()}
    if (
        any(
            values.ndim != 1 or not 1 <= len(values) <= 10000 or not np.isfinite(values).all()
            for values in arrays.values()
        )
        or len({len(values) for values in arrays.values()}) != 1
    ):
        raise ValueError("Return equal-length finite numeric columns, with 1–10000 rows")
    return {"columns": {key: values.tolist() for key, values in arrays.items()}, "units": units}


if __name__ == "__main__":
    try:
        print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2])), allow_nan=False))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
