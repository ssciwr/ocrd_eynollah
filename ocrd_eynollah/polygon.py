from shapely.geometry import LineString, Polygon
from shapely.ops import orient


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
    bridge = LineString([shell_point, hole_point])
    if bridge.length == 0 or not bridge.covered_by(poly):
        return False

    boundary_intersection = bridge.intersection(poly.boundary)
    return boundary_intersection.length == 0


def _find_hole_bridge(
    poly: Polygon, shell: list[tuple[int, int]], hole: list[tuple[int, int]]
) -> tuple[int, int]:
    candidates = []
    for shell_idx, shell_point in enumerate(shell):
        for hole_idx, hole_point in enumerate(hole):
            dist2 = (shell_point[0] - hole_point[0]) ** 2 + (
                shell_point[1] - hole_point[1]
            ) ** 2
            candidates.append((dist2, shell_idx, hole_idx))

    for _, shell_idx, hole_idx in sorted(candidates):
        if _bridge_is_inside_polygon(poly, shell[shell_idx], hole[hole_idx]):
            return shell_idx, hole_idx

    # could not find a bridge from polygon exterior to interior
    return None, None


def _rotate_ring(
    points: list[tuple[int, int]], start_idx: int
) -> list[tuple[int, int]]:
    return points[start_idx:] + points[:start_idx]


def _valid_ring(ring: list[tuple[int, int]]) -> bool:
    return len(set(ring)) >= 3


def _polygon_holes(poly: Polygon) -> list[list[tuple[int, int]]]:
    holes = []
    for interior in poly.interiors:
        hole = _integer_ring_points(interior.coords)
        if _valid_ring(hole):
            holes.append(hole)
    return holes


def _cut_open_polygon_points(poly: Polygon, holes: list[list[tuple[int, int]]]
                             ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
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
    backlog_holes: list[list[tuple[int, int]]] = []

    for hole in holes:
        if not _valid_ring(hole):
            continue

        shell_idx, hole_idx = _find_hole_bridge(poly, shell, hole)
        if shell_idx is None or hole_idx is None:
            backlog_holes.append(hole)
            continue

        bridges.setdefault(shell_idx, []).append(_rotate_ring(hole, hole_idx))

    points: list[tuple[int, int]] = []
    for shell_idx, shell_point in enumerate(shell):
        points.append(shell_point)
        for hole in bridges.get(shell_idx, []):
            hole_start = hole[0]
            points.append(hole_start)
            points.extend(hole[1:])
            points.append(hole_start)
            points.append(shell_point)

    return points, backlog_holes


def cut_open_polygon(poly: Polygon,
                     holes: list[list[tuple[int, int]]] | None = None,
                     max_depth: int = 5) -> list[tuple[int, int]]:
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
       - ``max_depth`` recursive passes have been attempted.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    
    holes = holes if holes is not None else _polygon_holes(poly)
    
    if not holes:
        shell = _integer_ring_points(orient(poly, sign=1.0).exterior.coords)
        return shell if _valid_ring(shell) else []
    
    points, backlog_holes = _cut_open_polygon_points(poly, holes)

    # no valid shell could be constructed, 
    # or all holes have been cut open,
    # or we have reached maximum recursion depth
    if not points or not backlog_holes or max_depth == 1:
        return points
    
    # rebuild polygon and retry remaining holes
    new_poly = Polygon(points)
    if new_poly.is_empty:
        return points
    
    return cut_open_polygon(new_poly, backlog_holes, max_depth - 1)



def page_points_from_polygon(poly: Polygon) -> str:
    """Convert a Shapely polygon to a PAGE coordinate sequence."""
    return " ".join("%i,%i" % point for point in cut_open_polygon(poly, holes=None, max_depth=5))
