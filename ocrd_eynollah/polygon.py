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

    raise ValueError("Could not find a bridge from polygon exterior to interior")


def _rotate_ring(
    points: list[tuple[int, int]], start_idx: int
) -> list[tuple[int, int]]:
    return points[start_idx:] + points[:start_idx]


def cut_open_polygon_points(poly: Polygon) -> list[tuple[int, int]]:
    """Return PAGE points for ``poly`` with holes represented by duplicate cuts.

    PAGE XML has a single polygon coordinate sequence per region. For polygons
    with interiors, serialize each interior as a detour from the exterior ring:
    exterior point -> interior point -> interior ring -> interior point ->
    exterior point. The bridge edge is therefore present twice in opposite
    directions and contributes no area.
    """
    poly = orient(poly, sign=1.0)
    shell = _integer_ring_points(poly.exterior.coords)
    if len(set(shell)) < 3:
        return []

    bridges: dict[int, list[list[tuple[int, int]]]] = {}
    for interior in poly.interiors:
        hole = _integer_ring_points(interior.coords)
        if len(set(hole)) < 3:
            continue

        shell_idx, hole_idx = _find_hole_bridge(poly, shell, hole)
        bridges.setdefault(shell_idx, []).append(_rotate_ring(hole, hole_idx))

    points = []
    for shell_idx, shell_point in enumerate(shell):
        points.append(shell_point)
        for hole in bridges.get(shell_idx, []):
            hole_start = hole[0]
            points.append(hole_start)
            points.extend(hole[1:])
            points.append(hole_start)
            points.append(shell_point)

    return points


def page_points_from_polygon(poly: Polygon) -> str:
    """Convert a Shapely polygon to a PAGE coordinate sequence."""
    return " ".join("%i,%i" % point for point in cut_open_polygon_points(poly))
