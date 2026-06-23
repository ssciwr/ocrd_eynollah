from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import orient
from dataclasses import dataclass
import heapq
from scipy.spatial import cKDTree
import numpy as np


def _integer_ring_points(coords) -> list[tuple[int, int]]:
    points = []
    for coord in coords:
        x, y = coord[:2]
        point = (int(round(x)), int(round(y)))
        if not points or points[-1] != point:
            points.append(point)

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()

    return points


def _bridge_is_inside_polygon(poly: Polygon, shell_point, hole_point) -> bool:
    if shell_point == hole_point:
        return False

    bridge = LineString([shell_point, hole_point])

    if bridge.length == 0 or not bridge.covered_by(poly):
        return False

    # avoid using intersection to reduce runtime
    # remove the endpoints so touching the shell/hole boundary at endpoints
    # is allowed, but the interior of the bridge may not touch/cross boundary.
    eps = min(0.5, bridge.length * 1e-6)
    if bridge.length <= 2 * eps:
        return False

    inner = LineString(
        [
            bridge.interpolate(eps),
            bridge.interpolate(bridge.length - eps),
        ]
    )

    return poly.covers(inner) and not inner.intersects(poly.boundary)


def _find_hole_bridge(
    poly: Polygon,
    shell: list[tuple[int, int]],
    hole: list[tuple[int, int]],
    shell_k: int = 8,
) -> tuple[int | None, int | None]:
    """Find a valid bridge between shell and hole.

    Instead of testing every shell x hole pair, only consider the k nearest
    shell vertices for each hole vertex, then test all those candidates in
    increasing distance order.
    """
    if not shell or not hole:
        return None, None

    shell_arr = np.asarray(shell, dtype=float)
    hole_arr = np.asarray(hole, dtype=float)

    tree = cKDTree(shell_arr)
    k = min(shell_k, len(shell))

    dists, idxs = tree.query(hole_arr, k=k)

    # normalize shapes for k=1
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    candidate_heap: list[tuple[float, int, int]] = []
    for hole_idx in range(len(hole)):
        for j in range(k):
            shell_idx = int(idxs[hole_idx, j])
            dist2 = float(dists[hole_idx, j]) ** 2
            candidate_heap.append((dist2, shell_idx, hole_idx))

    heapq.heapify(candidate_heap)

    while candidate_heap:
        _, shell_idx, hole_idx = heapq.heappop(candidate_heap)
        if _bridge_is_inside_polygon(poly, shell[shell_idx], hole[hole_idx]):
            return shell_idx, hole_idx

    # could not find a bridge from polygon exterior to interior
    return None, None


def _rotate_ring(
    points: list[tuple[int, int]], start_idx: int
) -> list[tuple[int, int]]:
    return points[start_idx:] + points[:start_idx]


def _valid_ring(ring: list[tuple[int, int]]) -> bool:
    """Return True if the ring has at least 3 unique points.
    Avoid using len of a large ring.
    """
    seen = set()
    for p in ring:
        seen.add(p)
        if len(seen) >= 3:
            return True
    return False


@dataclass(slots=True)
class HoleData:
    points: list[tuple[int, int]]
    area: float


def _polygon_holes(poly: Polygon, min_area: float = 100.0) -> list[HoleData]:
    holes: list[HoleData] = []
    for interior in poly.interiors:
        hole = _integer_ring_points(interior.coords)
        if not _valid_ring(hole):
            continue

        area = Polygon(interior).area
        if area >= min_area:
            holes.append(HoleData(points=hole, area=area))

    return holes


def _cut_open_polygon_points(
    poly: Polygon, holes: list[HoleData]
) -> tuple[list[tuple[int, int]], list[HoleData]]:
    """Return PAGE points for ``poly`` with holes represented by duplicate cuts.

    PAGE XML has a single polygon coordinate sequence per region. For polygons
    with interiors, serialize each interior as a detour from the exterior ring:
    exterior point -> interior point -> interior ring -> interior point ->
    exterior point. The bridge edge is therefore present twice in opposite
    directions and contributes no area.

    This helper function performs one round over the supplied holes:
    - holes for which a valid bridge can be found are inserted into the shell,
    - holes for which no valid bridge can be found are returned in the backlog
      for another round.

    The function does not recurse itself; it performs one cutting pass. The
    public ``cut_open_polygon()`` function repeatedly rebuilds the polygon and
    retries unresolved holes until all holes are cut open or no further
    progress can be made.
    """
    poly = orient(poly, sign=1.0)
    shell = _integer_ring_points(poly.exterior.coords)
    if not _valid_ring(shell):
        return [], holes

    bridges: dict[int, list[list[tuple[int, int]]]] = {}
    backlog_holes: list[HoleData] = []

    for hole in holes:
        shell_idx, hole_idx = _find_hole_bridge(poly, shell, hole.points)
        if shell_idx is None or hole_idx is None:
            backlog_holes.append(hole)
            continue

        bridges.setdefault(shell_idx, []).append(_rotate_ring(hole.points, hole_idx))

    points: list[tuple[int, int]] = []
    append = points.append
    extend = points.extend
    get_bridges = bridges.get

    for shell_idx, shell_point in enumerate(shell):
        append(shell_point)
        shell_bridges = get_bridges(shell_idx)
        if not shell_bridges:
            continue

        for h in shell_bridges:
            hole_start = h[0]
            append(hole_start)
            extend(h[1:])
            append(hole_start)
            append(shell_point)

    return points, backlog_holes


def cut_open_polygon(
    poly: Polygon,
    holes: list[HoleData] | None = None,
    max_loop: int = 3,
    backlog_area_ratio_threshold: float | None = 0.01,
    min_area: float = 100.0,
) -> list[tuple[int, int]]:
    """Return a new PAGE polygon with holes cut open as duplicate bridges.

    The function converts a polygon with interiors into a single coordinate
    sequence suitable for PAGE XML by repeatedly cutting holes into the shell
    using duplicate bridge edges.

    Algorithm:
    1. Start from the polygon's exterior ring and the set of hole rings.
    2. In one pass, cut every hole for which a valid shell-to-hole bridge can
       be found.
    3. Rebuild a polygon from the resulting flattened point sequence.
    4. Retry any holes that could not be bridged in the previous pass.
    5. Stop when either:
       - all holes have been cut open, or
       - a pass makes no progress / yields an invalid shell, or
       - the ratio of unresolved hole area to polygon area is below the threshold, or
       - ``max_loop`` runs have been attempted.
    """
    if max_loop < 1:
        raise ValueError("max_loop must be at least 1")

    poly = orient(poly, sign=1.0)

    total_poly_area = poly.area

    # assume that all holes are valid and have been filtered by min_area
    holes = holes if holes is not None else _polygon_holes(poly, min_area=min_area)

    if not holes:
        shell = _integer_ring_points(poly.exterior.coords)
        return shell if _valid_ring(shell) else []

    current_poly = poly
    current_holes = holes
    points: list[tuple[int, int]] = []

    for _ in range(max_loop):
        points, backlog_holes = _cut_open_polygon_points(current_poly, current_holes)

        # invalid shell
        if not points:
            return []

        # all holes resolved
        if not backlog_holes:
            return points

        # early stop if unsolved holes are too small relative to the polygon area
        if backlog_area_ratio_threshold is not None and total_poly_area > 0:
            backlog_area = sum(h.area for h in backlog_holes)
            if backlog_area / total_poly_area < backlog_area_ratio_threshold:
                return points

        # rebuild polygon from flattened points and retry unresolved holes
        new_poly = Polygon(points)
        if new_poly.is_empty:
            return points

        current_poly = new_poly
        current_holes = backlog_holes

    return points


def page_points_from_polygon(
    poly: Polygon,
    max_loop: int = 3,
    backlog_area_ratio_threshold: float | None = 0.01,
    min_area: float = 100.0,
) -> str:
    """Convert a Shapely polygon to a PAGE coordinate sequence."""
    points = cut_open_polygon(
        poly,
        holes=None,
        max_loop=max_loop,
        backlog_area_ratio_threshold=backlog_area_ratio_threshold,
        min_area=min_area,
    )
    return " ".join(f"{x},{y}" for x, y in points)


def flatten_polygon_geometry(geometry) -> list[Polygon]:
    """Return all polygon components from a Shapely geometry."""
    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        result = []
        for geom in geometry.geoms:
            result.extend(flatten_polygon_geometry(geom))
        return result

    return []
