from shapely.geometry import Polygon

from ocrd_eynollah.polygon import (
    cut_open_polygon_points,
    page_points_from_polygon,
)


def _signed_area(points):
    return (
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        / 2
    )


def test_cut_open_polygon_preserves_area_and_connects_hole():
    polygon = Polygon(
        shell=[(0, 0), (20, 0), (20, 20), (0, 20)],
        holes=[[(5, 5), (5, 15), (15, 15), (15, 5)]],
    )

    points = cut_open_polygon_points(polygon)
    page_points = [
        tuple(map(int, point.split(",")))
        for point in page_points_from_polygon(polygon).split()
    ]

    assert page_points == points
    assert abs(_signed_area(points)) == polygon.area
    assert len(points) == 10

    repeated_points = {point for point in points if points.count(point) > 1}
    assert len(repeated_points) == 2
