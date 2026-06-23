from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from ocrd_eynollah.polygon import flatten_polygon_geometry


def overlay_outline(
    image: Image.Image, result: dict[str, list[Polygon]]
) -> Image.Image:
    """ "Overlay the detected polygons on the original image for visualization.

    Args:
        image (Image.Image): original image to overlay on
        result (dict[str, list[Polygon]]): dictionary containing the detected polygons,
            with possible keys:
                "artificial_boundary", "text", "image", "heading", and "separator"

    Returns:
        Image.Image: image with overlaid polygons.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGBA")

    def _draw_polygons(polys, line_color, width=4, fill_color=None):
        for poly in polys:
            poly_iter = flatten_polygon_geometry(poly)

            for p in poly_iter:
                # create a grayscale mask for the polygon
                mask = Image.new("L", image.size, 0)
                mask_draw = ImageDraw.Draw(mask)

                # draw outer polygon
                mask_draw.polygon(
                    p.exterior.coords,
                    fill=255,  # visible area
                )

                # cut out holes
                for hole in p.interiors:
                    mask_draw.polygon(
                        hole.coords,
                        fill=0,  # invisible area
                    )

                # apply the fill color using mask
                if fill_color is not None:
                    r, g, b, a = fill_color
                    base_layer = Image.new("RGBA", image.size, (r, g, b, 0))
                    alpha_mask = mask.point(lambda p: int(p * (a / 255)))
                    base_layer.putalpha(alpha_mask)
                    image.alpha_composite(base_layer)

                # draw the border on top of the fill
                border_draw = ImageDraw.Draw(image, "RGBA")

                border_draw.polygon(
                    p.exterior.coords,
                    outline=line_color,
                    width=width,
                    fill=None,  # no fill for border, only outline
                )

                for hole in p.interiors:
                    border_draw.polygon(
                        hole.coords,
                        outline=line_color,
                        width=width,
                        fill=None,  # no fill for border, only outline
                    )

    # fill color with alpha for better visualization of overlaps
    fill_color_artificial_boundary = (0, 204, 0, 77)
    border_color_artificial_boundary = (0, 204, 0, 255)
    fill_color_text = (231, 76, 60, 77)
    border_color_text = (231, 76, 60, 255)
    fill_color_image = (52, 152, 219, 77)
    border_color_image = (52, 152, 219, 255)
    fill_color_heading = (230, 126, 34, 77)
    border_color_heading = (230, 126, 34, 255)
    fill_color_separator = (155, 89, 182, 77)
    border_color_separator = (155, 89, 182, 255)

    # draw in priority order
    # according to order from LabelStudio annotation
    # from low to highter hierarchy level:
    # artificial_boundary -> text -> image -> heading -> separator

    # artificial boundary polygons (green)
    if "artificial_boundary" in result:
        _draw_polygons(
            result["artificial_boundary"],
            border_color_artificial_boundary,
            width=4,
            fill_color=fill_color_artificial_boundary,
        )
    # text polygons (red)
    if "text" in result:
        _draw_polygons(
            result["text"], border_color_text, width=4, fill_color=fill_color_text
        )
    # image polygons (blue)
    if "image" in result:
        _draw_polygons(
            result["image"],
            border_color_image,
            width=4,
            fill_color=fill_color_image,
        )
    # heading polygons (yellow)
    if "heading" in result:
        _draw_polygons(
            result["heading"],
            border_color_heading,
            width=4,
            fill_color=fill_color_heading,
        )
    # separator polygons (purple)
    if "separator" in result:
        _draw_polygons(
            result["separator"],
            border_color_separator,
            width=4,
            fill_color=fill_color_separator,
        )

    return image
