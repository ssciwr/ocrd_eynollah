from shapely.geometry import Polygon

from ocrd_eynollah.polygon import (
    cut_open_polygon,
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

    points = cut_open_polygon(polygon)
    page_points = [
        tuple(map(int, point.split(",")))
        for point in page_points_from_polygon(polygon).split()
    ]

    assert page_points == points
    assert abs(_signed_area(points)) == polygon.area
    assert len(points) == 10

    repeated_points = {point for point in points if points.count(point) > 1}
    assert len(repeated_points) == 2


def test_cut_open_polygon_with_multiple_holes():
    # polygon with 1 small hole in the center and 8 surrounding big holes
    polygon = Polygon(
        shell=[(0, 0), (100, 0), (100, 100), (0, 100)],
        holes=[
            [(10, 10), (10, 30), (30, 30), (30, 10)],
            [(70, 10), (70, 30), (90, 30), (90, 10)],
            [(10, 70), (10, 90), (30, 90), (30, 70)],
            [(70, 70), (70, 90), (90, 90), (90, 70)],
            [(45, 45), (45, 55), (55, 55), (55, 45)],
            [(35, 10), (35, 30), (65, 30), (65, 10)],
            [(10, 35), (10, 65), (30, 65), (30, 35)],
            [(70, 35), (70, 65), (90, 65), (90, 35)],
            [(35, 70), (35, 90), (65, 90), (65, 70)],
        ],
    )

    # ignore the center hole
    points = cut_open_polygon(
        polygon,
        holes=None,
        max_loop=3,
        backlog_area_ratio_threshold=0.01,
        min_area=200.0,
    )
    page_points = [
        tuple(map(int, point.split(",")))
        for point in page_points_from_polygon(
            polygon, max_loop=3, backlog_area_ratio_threshold=0.01, min_area=200.0
        ).split()
    ]

    assert page_points == points
    assert abs(_signed_area(points)) == polygon.area + 100  # area of the center hole
    assert len(points) == 52

    repeated_points = {point for point in points if points.count(point) > 1}
    assert len(repeated_points) == 12

    # include the center hole
    points = cut_open_polygon(
        polygon,
        holes=None,
        max_loop=3,
        backlog_area_ratio_threshold=0.01,
        min_area=100.0,
    )

    assert abs(_signed_area(points)) == polygon.area
    assert len(points) == 58

    repeated_points = {point for point in points if points.count(point) > 1}
    assert len(repeated_points) == 14
