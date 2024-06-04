from itertools import tee
import numpy as np
np.random.seed(0)

from shapely.geometry import Point, GeometryCollection, Polygon

def _flat_hex_coords(center, radius, i):
    """Return the point coordinate of a flat-topped regular hexagon.
    points are returned in counter-clockwise order as i increases
    the first coordinate (i=0) will be:
    center.x + size, center.y

    """
    #angle_deg = 60 * i
    angle_rad = np.deg2rad(60 * i)
    return Point(
        center[0] + radius * np.cos(angle_rad),
        center[1] + radius * np.sin(angle_rad),
        center[2]
    )

#hexagon = [_flat_hex_coords(center, radius, i) for i in range(6)]
def make_hexagonal_polygons(self, centers = None, radius = None):
    """
    """
    hexagons = []
    side_points = []
    polygons = []
    if centers is None:
        centers = self.site_locations
        radius = self.cell_radius
    for center in centers:
        hexagon = [_flat_hex_coords(center, radius, i) for i in range(6)]
        tuple_points = zip(hexagon, hexagon[1:] + [hexagon[0]])
        for pair in tuple_points:
            if list(pair) not in side_points:
                side_points.append(list(pair))
        hexagons.append(hexagon)
        for poly in hexagons:
            if Polygon(poly) not in polygons:
                polygons.append(Polygon(poly))
    return hexagons, side_points, polygons

# 
def pairwise(iterable):
    """

    Return iterable of 2-tuples in a sliding window

    Parameters
    ----------
    iterable: list
        Sliding window

    Returns
    -------
    list of tuple
        Iterable of 2-tuples

    Example
    -------
        >>> list(pairwise([1,2,3,4]))
            [(1,2),(2,3),(3,4)]

    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# A simple function to make sure a value is in a given range, -1 to 1 by
# default
def clamp(i, mn=-1, mx=1):
    return min(max(i, mn), mx)

