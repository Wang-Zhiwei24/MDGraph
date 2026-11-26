"""Read-only discovery of elemental EAM setfl tables in NIST's public repository."""

from __future__ import annotations

import csv
import io
import math
import re
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ORIGIN = "https://www.ctcms.nist.gov"


def checked_url(url: str) -> str:
    p = urlsplit(url)
    if (
        p.scheme != "https"
        or p.netloc not in {"www.ctcms.nist.gov", "ctcms.nist.gov"}
        or not re.fullmatch(r"/potentials/(system|entry|Download)/[A-Za-z0-9_.\-/]+", p.path)
        or ".." in p.path.split("/")
        or p.query
        or p.fragment
    ):
        raise ValueError("URL is outside the supported NIST repository paths")
    return url


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(url: str, limit: int) -> bytes:
    opener = build_opener(ProxyHandler({}), NoRedirects())
    request = Request(checked_url(url), headers={"User-Agent": "MDGraph NIST materials lookup"})
    for attempt in range(2):
        try:
            with opener.open(request, timeout=20) as response:
                data = response.read(limit + 1)
            break
        except HTTPError as error:
            if attempt or error.code < 500:
                raise
        except OSError:
            if attempt:
                raise
    if len(data) > limit:
        raise ValueError("NIST resource exceeds the configured size limit")
    return data


class IndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.stack = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "li":
            self.stack.append({"text": [], "links": []})
        if tag == "a" and self.stack and attrs.get("href"):
            self.stack[-1]["links"].append(attrs["href"])
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag == "li" and self.stack:
            self.items.append(self.stack.pop())
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if self.stack and not self.hidden:
            self.stack[-1]["text"].append(data)


def index_candidates(url: str, html: bytes) -> list[dict]:
    parser = IndexParser()
    parser.feed(html.decode("utf-8", errors="replace"))
    candidates = {}
    for item in parser.items:
        label = " ".join(" ".join(item["text"]).split())
        if not re.match(r"LAMMPS pair_style eam/alloy\b", label, re.I):
            continue
        if re.search(r"file\(s\):\s*(?:superseded|retracted)", label, re.I):
            continue
        links = []
        for href in item["links"]:
            try:
                links.append(checked_url(urljoin(url, href)))
            except ValueError:
                continue
        site = [x for x in links if "/entry/" in x and re.search(r"--LAMMPS--ipr\d+\.html$", x)]
        files = [
            x
            for x in links
            if "/Download/" in x and x.lower().endswith((".eam.alloy", ".eam", ".set", ".eamalloy"))
        ]
        if len(set(site)) == 1 and len(set(files)) == 1:
            key = site[0].rsplit("/", 1)[-1].removesuffix(".html")
            candidates[key] = {"id": key, "kind": "eam/alloy", "source": site[0], "url": files[0]}
    return [candidates[key] for key in sorted(candidates)]


def crystal_evidence(data: bytes, phase: str) -> dict | None:
    prototype = {"fcc": "A1--Cu--fcc", "bcc": "A2--W--bcc"}[phase]
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
        if row.get("prototype") != prototype or (row.get("transformed") or "").lower() != "false":
            continue
        try:
            lattice = [float(row[k]) for k in ("a", "b", "c")]
            angles = [float(row[k]) for k in ("alpha", "beta", "gamma")]
        except (KeyError, ValueError, TypeError):
            continue
        if all(
            math.isfinite(v) and v > 0 and math.isclose(v, lattice[0], rel_tol=1e-5)
            for v in lattice
        ) and all(math.isclose(v, 90, abs_tol=1e-3) for v in angles):
            return {
                "prototype": prototype,
                "lattice_angstrom": lattice,
                "method": row.get("method"),
                "meaning": "NIST computed result for this model, not independent experimental validation",
            }
    return None


def validate_setfl(data: bytes, element: str) -> None:
    lines = data.decode("ascii", errors="strict").splitlines()
    try:
        if lines[3].split() != ["1", element]:
            raise ValueError("Wrong element list")
        nrhos, drho, nrs, dr, cutoff = lines[4].split()
        nrho, nr = int(nrhos), int(nrs)
        if not (
            4 <= nrho <= 1000000
            and 4 <= nr <= 1000000
            and all(math.isfinite(x) and x > 0 for x in map(float, (drho, dr, cutoff)))
        ):
            raise ValueError("Invalid grid")
        from ase.data import atomic_numbers

        atom, mass, alat, _ = lines[5].split()
        if not (
            int(atom) == atomic_numbers[element]
            and 0 < float(mass) < 300
            and 0 <= float(alat) < 100
        ):
            raise ValueError("Invalid element header")
        values = [float(x) for line in lines[6:] for x in line.split()]
        if len(values) != nrho + 2 * nr or not all(map(math.isfinite, values)):
            raise ValueError("Invalid table values")
    except (IndexError, ValueError, KeyError) as exc:
        raise ValueError(
            "NIST download is not a valid single-element EAM/alloy table for this element"
        ) from exc


def elastic_evidence(data: bytes, element: str, phase: str) -> dict | None:
    prototype = {"fcc": "A1--Cu--fcc", "bcc": "A2--W--bcc"}[phase]
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
        if row.get("composition") != element or row.get("prototype") != prototype:
            continue
        try:
            values = [float(row[key]) for key in ("C11", "C12", "C44")]
        except (ValueError, KeyError, TypeError):
            continue
        if (
            all(map(math.isfinite, values))
            and min(values[0] - values[1], values[0] + 2 * values[1], values[2]) > 0
        ):
            return {
                "prototype": prototype,
                "nist_computed": True,
                "meaning": "NIST has a computed elastic tensor for this phase; this is not experimental validation",
            }
    return None


def discover(
    element: str, phase: str, target_property: str = "lattice_constant"
) -> tuple[dict, list[dict]]:
    if not re.fullmatch(r"[A-Z][a-z]?", element):
        raise ValueError("Invalid element for NIST discovery")
    url = f"{ORIGIN}/potentials/system/{element}/"
    try:
        raw = fetch(url, 2_000_000)
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"NIST does not list this element system: {element}") from exc
        raise
    candidates = index_candidates(url, raw)
    if not candidates:
        raise ValueError("NIST index returned no qualifying EAM/alloy implementation")

    def check(c):
        elastic = target_property == "elastic_constants"
        prefix = "elastic" if elastic else "crystal"
        table = c["source"].removesuffix(".html") + f"/{prefix}.{element}.csv"
        try:
            raw_table = fetch(table, 2_000_000)
            row = (
                elastic_evidence(raw_table, element, phase)
                if elastic
                else crystal_evidence(raw_table, phase)
            )
            if row is None:
                return c["id"], "No matching cubic-phase evidence in the NIST table", None
            if elastic:
                # Verify this is a true single-element file before presenting an executable choice.
                model = fetch(c["url"], 12_000_000)
                validate_setfl(model, element)
                c = {**c, "discovery_model_sha256": sha256(model).hexdigest()}
            return c["id"], "supported", {**c, "evidence_url": table, "evidence": row}
        except (OSError, ValueError) as error:
            return c["id"], f"{type(error).__name__}: {error}", None

    # Bound the number of external queries/candidates. This shortlist does not claim exhaustive coverage.
    examined = candidates[:20]
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(check, examined))
    supported = [c for _, _, c in checked if c]
    info = {
        "index_url": url,
        "index_sha256": sha256(raw).hexdigest(),
        "index_candidates": len(candidates),
        "examined": {k: reason for k, reason, _ in checked},
        "eligible_count": len(supported),
        "shortlist_limit": 6,
        "shortlist_policy": "First six evidence-supported implementations in sorted implementation-ID order",
    }
    return info, supported[:6]
