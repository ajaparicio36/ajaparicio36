"""Generate the profile README artwork into ../assets.

GitHub strips CSS and JS from a README and serves repo SVGs with a strict
content policy (no fonts, no nested images), so everything here is pure
vector: text is outlined from the Allo variable font and the only motion is
SVG-native.

Run:  python tools/generate.py
Needs: numpy, fonttools, brotli, uharfbuzz
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "assets"

# The blue scale from the portfolio.
GROUND = "#02040d"
RAISED = "#060b1c"
ABYSS = "#03195b"
ROYAL = "#0b2cb1"
ELECTRIC = "#1951fc"
AZURE = "#3781fc"
SKY = "#62bbfd"
ICE = "#cbe9fd"
TEXT = "#eaf5ff"
MUTED = "#a3bcdd"

# ---------------------------------------------------------------- content

NAME = "Anthony John Aparicio"
ROLE = "Full-stack Engineer"

LINKS = [
    ("portfolio", "Portfolio ↗", True),
    ("linkedin", "LinkedIn ↗", False),
    ("email", "Email ↗", False),
    ("resume", "Resume ↗", False),
]

LANGUAGES = ["TypeScript", "Go", "Java", "Python", "C#", "C++", "Rust"]

SKILL_GROUPS = [
    ("Backend and architecture",
     ["REST API", "GraphQL", "NestJS", "FastAPI", "Fastify", "Express.js", "Gin", ".NET"]),
    ("Frontend and mobile",
     ["Next.js", "Expo React Native", "Flutter", "Tailwind CSS", "Angular", "Vue", "WPF UI"]),
    ("Database",
     ["PostgreSQL", "MongoDB", "Prisma ORM", "Firebase"]),
    ("DevOps and cloud",
     ["AWS", "Google Cloud", "Azure", "Linux", "Docker", "Kubernetes", "Terraform", "Vercel",
      "Railway"]),
    ("Automation and integrations",
     ["n8n", "Zoho", "Zapier", "Stripe", "PayMongo", "RevenueCat", "Shopify",
      "Google Maps and Auth APIs"]),
    ("AI, tools and systems",
     ["Claude Code", "Codex", "OpenCode", "MLKit", "YOLO26", "Raspberry Pi", "ESP32 / Arduino"]),
]

# ------------------------------------------------------------------- text


def fmt(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


class Type:
    """Shapes text with HarfBuzz and emits outlined glyphs as <use> refs."""

    def __init__(self, woff2: Path) -> None:
        buffer = io.BytesIO()
        font = TTFont(woff2)
        font.flavor = None  # HarfBuzz cannot read woff2
        font.save(buffer)
        self.face = hb.Face(buffer.getvalue())
        self.upem = self.face.upem
        self.defs: dict[tuple[int, int, int, int], tuple[str, str]] = {}

    def _font(self, wght: int, rond: int, opsz: int) -> hb.Font:
        font = hb.Font(self.face)
        font.set_variations({"wght": wght, "ROND": rond, "opsz": opsz})
        return font

    def _glyph(self, font: hb.Font, key: tuple[int, int, int, int]) -> str:
        if key not in self.defs:
            pen = SVGPathPen(None, ntos=lambda v: str(round(v)))
            font.draw_glyph_with_pen(key[0], pen)
            self.defs[key] = (f"g{len(self.defs)}", pen.getCommands())
        return self.defs[key][0]

    def shape(self, text: str, wght: int, rond: int, opsz: int, tracking: float):
        font = self._font(wght, rond, opsz)
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(font, buf, {"kern": True, "ss03": True, "cv05": True})
        glyphs = []
        pen_x = 0.0
        extra = tracking * self.upem
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            ref = self._glyph(font, (info.codepoint, wght, rond, opsz))
            glyphs.append((ref, pen_x + pos.x_offset))
            pen_x += pos.x_advance + extra
        return glyphs, pen_x - extra

    def width(self, text: str, size: float, wght: int = 450, rond: int = 60,
              opsz: int = 14, tracking: float = 0.0) -> float:
        return self.shape(text, wght, rond, opsz, tracking)[1] * size / self.upem

    def draw(self, text: str, x: float, y: float, size: float, fill: str,
             wght: int = 450, rond: int = 60, opsz: int = 14, tracking: float = 0.0,
             anchor: str = "start", extra: str = "") -> str:
        """Baseline at (x, y). Returns an SVG group."""
        glyphs, advance = self.shape(text, wght, rond, opsz, tracking)
        scale = size / self.upem
        if anchor == "middle":
            x -= advance * scale / 2
        elif anchor == "end":
            x -= advance * scale
        uses = "".join(
            f'<use href="#{ref}" x="{round(gx)}"/>' if gx else f'<use href="#{ref}"/>'
            for ref, gx in glyphs
        )
        return (
            f'<g transform="translate({fmt(x)} {fmt(y)}) scale({scale:.5f} {-scale:.5f})" '
            f'fill="{fill}"{extra}>{uses}</g>'
        )

    def defs_markup(self) -> str:
        return "".join(f'<path id="{ref}" d="{d}"/>' for ref, d in self.defs.values())


# ---------------------------------------------------------------- terrain


def permute(x):
    return np.mod(((x * 34.0) + 1.0) * x, 289.0)


def snoise(vx, vy):
    """2D simplex noise, ported from the shader (Ian McEwan / Ashima, MIT)."""
    c0, c1, c2, c3 = 0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439
    s = (vx + vy) * c1
    ix, iy = np.floor(vx + s), np.floor(vy + s)
    t = (ix + iy) * c0
    x0x, x0y = vx - ix + t, vy - iy + t
    i1x = np.where(x0x > x0y, 1.0, 0.0)
    i1y = 1.0 - i1x
    x1x, x1y = x0x + c0 - i1x, x0y + c0 - i1y
    x2x, x2y = x0x + c2, x0y + c2
    ix, iy = np.mod(ix, 289.0), np.mod(iy, 289.0)
    total = 0.0
    for ox, oy, px, py in ((0.0, 0.0, x0x, x0y), (i1x, i1y, x1x, x1y), (1.0, 1.0, x2x, x2y)):
        p = permute(permute(iy + oy) + ix + ox)
        m = np.maximum(0.5 - (px * px + py * py), 0.0) ** 4
        x = 2.0 * np.modf(p * c3)[0] - 1.0
        h = np.abs(x) - 0.5
        a0 = x - np.floor(x + 0.5)
        m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h))
        total = total + m * (a0 * px + h * py)
    return 130.0 * total


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Terrain:
    """The same height field as the site's shader, frozen at one moment."""

    def __init__(self, valley: float, amp: float, time: float) -> None:
        self.valley, self.amp, self.time = valley, amp, time

    def height(self, x, y):
        v, t = self.valley, self.time
        sgn = np.where(x >= 0.0, 1.0, -1.0)
        side = smoothstep(v * 0.2, v * 1.3, np.abs(x))
        flow = t * 0.035
        n = snoise(x * 0.09 + sgn * 3.7, y * 0.075 - flow) * 0.5 + 0.5
        swell = 0.8 + 0.3 * np.sin(y * 0.2 + sgn * 1.2 + t * 0.05 * sgn)
        ridge = side * side * (2.9 + 3.4 * n) * swell
        dx, dy = x - v * 0.55, y - 1.5
        dune = 0.42 * np.exp(-(dx * dx * 0.45 + dy * dy * 0.16))
        ripple = 0.07 * snoise(x * 0.4, y * 0.3 - flow * 2.0)
        return ridge * self.amp + dune + ripple


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = points[b] - points[a]
        norm = math.hypot(seg[0], seg[1]) or 1.0
        rel = points[a + 1:b] - points[a]
        dist = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / norm
        i = int(np.argmax(dist))
        if dist[i] > tolerance:
            mid = a + 1 + i
            keep[mid] = True
            stack += [(a, mid), (mid, b)]
    return points[keep]


def ridgeline_paths(width: int, height: int, *, valley: float, amp: float, time: float,
                    cam=(0.0, 1.25, 9.0), look=(0.0, 2.55, 0.0), fov: float = 46.0,
                    spacing: float = 0.085, bands: int = 5) -> list[str]:
    """Project the stripes, drop what the terrain hides, bucket by depth."""
    terrain = Terrain(valley, amp, time)
    cam_v, look_v = np.array(cam), np.array(look)
    forward = look_v - cam_v
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal = (height / 2) / math.tan(math.radians(fov) / 2)

    xs = np.arange(-15.0, 15.0 + 1e-6, spacing)
    ys = np.linspace(-cam[2] + 0.35, 15.0, 260)  # local y, away from the camera
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    world = np.stack([gx, terrain.height(gx, gy), -gy], axis=-1)

    rel = world - cam_v
    depth = rel @ forward
    sx = width / 2 + (rel @ right) / np.maximum(depth, 1e-6) * focal
    sy = height / 2 - (rel @ up) / np.maximum(depth, 1e-6) * focal
    dist = np.linalg.norm(rel, axis=-1)

    visible = depth > 0.2
    for s in np.linspace(0.03, 0.985, 64):
        q = cam_v + rel * s
        visible &= terrain.height(q[..., 0], -q[..., 2]) <= q[..., 1] + 0.012
    margin = 30
    visible &= (sx > -margin) & (sx < width + margin) & (sy > -margin) & (sy < height + margin)

    edges = np.array([0.0, 6.5, 9.5, 13.0, 17.0, 1e9])[: bands + 1]
    edges[-1] = 1e9
    band = np.clip(np.searchsorted(edges, dist) - 1, 0, bands - 1)

    out: list[list[str]] = [[] for _ in range(bands)]
    for i in range(len(xs)):
        j = 0
        n = len(ys)
        while j < n:
            if not visible[i, j]:
                j += 1
                continue
            k = j
            while k + 1 < n and visible[i, k + 1] and band[i, k + 1] == band[i, j]:
                k += 1
            # Share the boundary point so adjacent depth bands join up.
            end = k + 1 if k + 1 < n and visible[i, k + 1] else k
            if end > j:
                pts = simplify(np.stack([sx[i, j:end + 1], sy[i, j:end + 1]], axis=1), 0.3)
                out[band[i, j]].append(
                    "M" + "L".join(f"{fmt(px)} {fmt(py)}" for px, py in pts)
                )
            j = k + 1
    return ["".join(segments) for segments in out]


# ------------------------------------------------------------------ parts

GRAIN = (
    '<filter id="grain" x="0" y="0" width="100%" height="100%">'
    '<feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed="7"/>'
    '<feColorMatrix values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 1.25 -0.32"/></filter>'
)

RIM = (
    '<linearGradient id="rim" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#fff" stop-opacity=".85"/>'
    f'<stop offset=".26" stop-color="{ICE}" stop-opacity=".2"/>'
    f'<stop offset=".62" stop-color="{AZURE}" stop-opacity=".16"/>'
    f'<stop offset="1" stop-color="{SKY}" stop-opacity=".75"/></linearGradient>'
)


def svg(width: float, height: float, defs: str, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(width)} {fmt(height)}" '
        f'width="{fmt(width)}" height="{fmt(height)}" role="img" aria-label="{title}">'
        f"<title>{title}</title><defs>{defs}</defs>{body}</svg>\n"
    )


def write(name: str, markup: str) -> None:
    path = OUT / name
    path.write_text(markup, encoding="utf-8", newline="\n")
    print(f"{name:24s} {path.stat().st_size / 1024:6.1f} KB")


# ----------------------------------------------------------------- banner


def banner() -> None:
    w, h, r = 1280, 440, 30
    kind = Type(ROOT / "font" / "AlloVariable.woff2")
    bands = ridgeline_paths(w, h, valley=5.9, amp=1.0, time=40.0,
                            cam=(0.0, 1.2, 9.0), look=(0.0, 2.2, 0.0), fov=40.0)
    widths = [1.5, 1.3, 1.1, 0.95, 0.8]
    base_alpha = [0.62, 0.52, 0.42, 0.3, 0.16]
    lit_alpha = [1.0, 0.95, 0.85, 0.65, 0.4]

    lines = "".join(
        f'<path id="b{i}" d="{d}" stroke-width="{widths[i]}"/>' for i, d in enumerate(bands)
    )
    base = "".join(
        f'<use href="#b{i}" stroke-opacity="{base_alpha[i]}"/>' for i in range(len(bands))
    )
    lit = "".join(
        f'<use href="#b{i}" stroke-opacity="{lit_alpha[i]}"/>' for i in range(len(bands))
    )

    name = kind.draw(NAME, w / 2, 168, 82, TEXT, wght=560, rond=100, opsz=32,
                     tracking=-0.035, anchor="middle")
    role = kind.draw(ROLE, w / 2, 214, 25, MUTED, wght=450, rond=60, opsz=24,
                     anchor="middle")

    defs = (
        f'<clipPath id="card"><rect width="{w}" height="{h}" rx="{r}"/></clipPath>'
        + GRAIN + RIM
        # The light: an ice core falling off through the blue scale.
        + '<radialGradient id="lit" gradientUnits="userSpaceOnUse" cx="1040" cy="270" r="600">'
        f'<stop offset="0" stop-color="#fff"/><stop offset=".1" stop-color="{ICE}"/>'
        f'<stop offset=".22" stop-color="{SKY}"/>'
        f'<stop offset=".4" stop-color="{AZURE}"/><stop offset=".6" stop-color="{ELECTRIC}"/>'
        f'<stop offset=".8" stop-color="{ROYAL}" stop-opacity=".7"/>'
        f'<stop offset="1" stop-color="{ABYSS}" stop-opacity="0"/>'
        '</radialGradient>'
        '<radialGradient id="glow" cx=".5" cy=".5" r=".5">'
        f'<stop offset="0" stop-color="{ELECTRIC}" stop-opacity=".5"/>'
        f'<stop offset=".45" stop-color="{ROYAL}" stop-opacity=".42"/>'
        f'<stop offset="1" stop-color="{ABYSS}" stop-opacity="0"/></radialGradient>'
        f'<linearGradient id="foot" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{GROUND}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{GROUND}" stop-opacity=".78"/></linearGradient>'
        '<filter id="halo" x="-30%" y="-80%" width="160%" height="260%">'
        '<feGaussianBlur stdDeviation="30"/></filter>'
        f'<g id="lines" fill="none" stroke-linecap="round" stroke-linejoin="round">{lines}</g>'
        f'<mask id="strokes" maskUnits="userSpaceOnUse" x="0" y="0" width="{w}" '
        f'height="{h}"><g fill="none" stroke="#fff" stroke-linecap="round" '
        f'stroke-linejoin="round">{lit}</g></mask>'
        '<style>@keyframes drift{0%,100%{transform:translate(0,0)}'
        '45%{transform:translate(-520px,26px)}80%{transform:translate(-140px,-12px)}}'
        '.drift{animation:drift 28s cubic-bezier(.45,0,.55,1) infinite}'
        '@media (prefers-reduced-motion:reduce){.drift{animation:none}}</style>'
        + kind.defs_markup()
    )
    body = (
        '<g clip-path="url(#card)">'
        f'<rect width="{w}" height="{h}" fill="{GROUND}"/>'
        # Sky light raking along the horizon from behind the right ridge.
        '<g class="drift"><ellipse cx="1020" cy="300" rx="560" ry="150" '
        'fill="url(#glow)" transform="rotate(-9 1020 300)"/></g>'
        f'<g fill="none" stroke-linecap="round" stroke-linejoin="round" stroke="{ROYAL}">'
        f"{base}</g>"
        # The lines mask a field of light that drifts behind them.
        '<g mask="url(#strokes)"><rect class="drift" x="-300" y="-400" width="2500" '
        'height="1300" fill="url(#lit)"/></g>'
        f'<rect y="{h - 150}" width="{w}" height="150" fill="url(#foot)"/>'
        # A pocket of dark sky so the name never sits on lit lines.
        f'<ellipse cx="{w / 2}" cy="166" rx="455" ry="78" fill="{GROUND}" opacity=".82" '
        'filter="url(#halo)"/>'
        f"{name}{role}"
        f'<rect width="{w}" height="{h}" filter="url(#grain)" opacity=".16"/>'
        "</g>"
        f'<rect x="1.25" y="1.25" width="{w - 2.5}" height="{h - 2.5}" rx="{r - 1}" fill="none" '
        'stroke="url(#rim)" stroke-width="2.5"/>'
    )
    # The band paths are referenced through <use>, so they live in defs.
    write("banner.svg", svg(w, h, defs, body, f"{NAME}, {ROLE}"))


# ------------------------------------------------------------------ links


def link(slug: str, label: str, primary: bool) -> None:
    kind = Type(ROOT / "font" / "AlloVariable.woff2")
    size, h, pad = 21, 56, 28
    text_w = kind.width(label, size, wght=600 if primary else 520)
    w = round(text_w + pad * 2)
    text = kind.draw(label, w / 2, h / 2 + 7.4, size, "#fff" if primary else TEXT,
                     wght=600 if primary else 520, anchor="middle")
    if primary:
        fill = (
            '<linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#4b79fd"/>'
            f'<stop offset=".46" stop-color="{ELECTRIC}"/>'
            '<stop offset="1" stop-color="#123ab5"/></linearGradient>'
        )
    else:
        fill = (
            '<linearGradient id="fill" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#16213f"/>'
            f'<stop offset=".5" stop-color="{RAISED}"/>'
            '<stop offset="1" stop-color="#0d1836"/></linearGradient>'
        )
    defs = (
        fill + RIM
        + '<radialGradient id="shine" cx=".14" cy="-.25" r=".9">'
        f'<stop offset="0" stop-color="#fff" stop-opacity="{".55" if primary else ".3"}"/>'
        '<stop offset=".5" stop-color="#fff" stop-opacity="0"/></radialGradient>'
        '<linearGradient id="floor" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset=".62" stop-color="{ICE}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{ICE if primary else SKY}" '
        f'stop-opacity="{".5" if primary else ".3"}"/></linearGradient>'
        + kind.defs_markup()
    )
    pill = f'width="{w - 2}" height="{h - 2}" x="1" y="1" rx="{(h - 2) / 2}"'
    body = (
        f'<rect {pill} fill="url(#fill)"/><rect {pill} fill="url(#floor)"/>'
        f'<rect {pill} fill="url(#shine)"/>{text}'
        f'<rect x="1.75" y="1.75" width="{w - 3.5}" height="{h - 3.5}" rx="{(h - 3.5) / 2}" '
        'fill="none" stroke="url(#rim)" stroke-width="2"/>'
    )
    write(f"link-{slug}.svg", svg(w, h, defs, body, label.replace(" ↗", "")))


# ----------------------------------------------------------------- skills


def skills() -> None:
    kind = Type(ROOT / "font" / "AlloVariable.woff2")
    w, pad, gap, r = 1280, 30, 16, 28
    lang_w = 320
    col_w = (w - pad * 2 - lang_w - gap * 2) / 2
    chip_size, chip_h, chip_pad, chip_gap = 19.5, 38, 15, 9
    cell_pad = 24

    def cell(title: str, items: list[str], x: float, y: float) -> tuple[str, float]:
        """Lay chips out in rows. Returns markup positioned at y=0 and height."""
        parts = [kind.draw(title, x + cell_pad, cell_pad + 19, 20, MUTED, wght=500)]
        cx, cy = x + cell_pad, cell_pad + 46
        for item in items:
            cw = kind.width(item, chip_size) + chip_pad * 2
            if cx + cw > x + col_w - cell_pad:
                cx, cy = x + cell_pad, cy + chip_h + chip_gap
            parts.append(
                f'<rect x="{fmt(cx)}" y="{fmt(cy)}" width="{fmt(cw)}" height="{chip_h}" '
                f'rx="{chip_h / 2}" fill="{ICE}" fill-opacity=".07" stroke="{ICE}" '
                'stroke-opacity=".2"/>'
            )
            parts.append(kind.draw(item, cx + chip_pad, cy + chip_h / 2 + 6.8, chip_size, TEXT))
            cx += cw + chip_gap
        return "".join(parts), cy + chip_h + cell_pad

    rows = []
    for i in range(0, len(SKILL_GROUPS), 2):
        pair = []
        for j, (title, items) in enumerate(SKILL_GROUPS[i:i + 2]):
            x = pad + lang_w + gap + j * (col_w + gap)
            pair.append((x, *cell(title, items, x, 0)))
        rows.append(pair)

    y = pad
    cells = []
    for pair in rows:
        row_h = max(height for _, _, height in pair)
        for x, markup, _ in pair:
            box = f'x="{fmt(x)}" y="0" width="{fmt(col_w)}" height="{fmt(row_h)}" rx="{r}"'
            cells.append(
                f'<g transform="translate(0 {fmt(y)})">'
                f'<rect {box} fill="url(#glass)"/><rect {box} fill="url(#sheen)"/>'
                f"{markup}"
                f'<rect x="{fmt(x + 1.25)}" y="1.25" width="{fmt(col_w - 2.5)}" '
                f'height="{fmt(row_h - 2.5)}" rx="{r - 1}" fill="none" stroke="url(#rim)" '
                'stroke-width="2.5"/></g>'
            )
        y += row_h + gap
    h = y - gap + pad

    lang_h = h - pad * 2
    langs = [kind.draw("Languages", pad + cell_pad, pad + cell_pad + 19, 20, ROYAL, wght=500)]
    line = 50
    start = pad + lang_h - cell_pad - (len(LANGUAGES) - 1) * line - 6
    for i, item in enumerate(LANGUAGES):
        langs.append(kind.draw(item, pad + cell_pad, start + i * line, 45, ABYSS, wght=600,
                               rond=100, opsz=32, tracking=-0.035))

    defs = (
        f'<clipPath id="card"><rect width="{w}" height="{fmt(h)}" rx="30"/></clipPath>'
        + GRAIN + RIM
        + '<filter id="soft" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="70"/></filter>'
        '<radialGradient id="lobe" cx=".5" cy=".5" r=".5">'
        f'<stop offset="0" stop-color="{SKY}" stop-opacity=".85"/>'
        f'<stop offset=".4" stop-color="{ELECTRIC}" stop-opacity=".8"/>'
        f'<stop offset=".75" stop-color="{ROYAL}" stop-opacity=".5"/>'
        f'<stop offset="1" stop-color="{ABYSS}" stop-opacity="0"/></radialGradient>'
        '<linearGradient id="glass" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{ICE}" stop-opacity=".17"/>'
        f'<stop offset=".42" stop-color="{AZURE}" stop-opacity=".07"/>'
        f'<stop offset="1" stop-color="{ICE}" stop-opacity=".1"/></linearGradient>'
        '<radialGradient id="sheen" cx=".08" cy="-.1" r=".9">'
        '<stop offset="0" stop-color="#fff" stop-opacity=".2"/>'
        '<stop offset=".5" stop-color="#fff" stop-opacity="0"/></radialGradient>'
        '<linearGradient id="lit" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#fff"/><stop offset=".5" stop-color="{ICE}"/>'
        f'<stop offset="1" stop-color="{SKY}"/></linearGradient>'
        + kind.defs_markup()
    )
    body = (
        '<g clip-path="url(#card)">'
        f'<rect width="{w}" height="{fmt(h)}" fill="{GROUND}"/>'
        '<g filter="url(#soft)" opacity=".55">'
        f'<ellipse cx="{w - 120}" cy="60" rx="420" ry="300" fill="url(#lobe)"/>'
        f'<ellipse cx="520" cy="{fmt(h + 40)}" rx="360" ry="230" fill="url(#lobe)" opacity=".7"/>'
        "</g>"
        f'<rect x="{pad}" y="{pad}" width="{lang_w}" height="{fmt(lang_h)}" rx="{r}" '
        'fill="url(#lit)"/>'
        f'<rect x="{pad + 0.75}" y="{pad + 0.75}" width="{lang_w - 1.5}" '
        f'height="{fmt(lang_h - 1.5)}" rx="{r - 0.5}" fill="none" stroke="#fff" '
        'stroke-opacity=".8" stroke-width="1.5"/>'
        + "".join(langs) + "".join(cells)
        + f'<rect width="{w}" height="{fmt(h)}" filter="url(#grain)" opacity=".1"/>'
        "</g>"
        f'<rect x="1.25" y="1.25" width="{w - 2.5}" height="{fmt(h - 2.5)}" rx="29" '
        'fill="none" stroke="url(#rim)" stroke-width="2.5" stroke-opacity=".7"/>'
    )
    summary = "Skills: " + ", ".join(LANGUAGES + [s for _, items in SKILL_GROUPS for s in items])
    write("skills.svg", svg(w, h, defs, body, summary))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    banner()
    for slug, label, primary in LINKS:
        link(slug, label, primary)
    skills()
