################################################################################
# proc/util.py
#
# Utilities that are useful when processing contours.

"""Contour processing utility functions."""


# Dependencies #################################################################

import numpy as np
import neuropythy as ny


# Functions ####################################################################

def extend_contour(pts, d=100):
    """Returns a copy of the given contour with both ends extended.

    `extend_contour(pts)` returns a copy of `pts` with a new point on each end.
    The new segments created by the two additional points are parallel with the
    previous end segments to which they are adjacent. The distance these points
    are extended can be controlled with the second parameter, `d` (default of
    100).
    """
    u = next(
        (d for ii in range(pts.shape[1] - 1)
         for d in [pts[:,ii] - pts[:,ii+1]]
         if np.sum(d**2) != 0))
    v = next(
        (d for ii in range(pts.shape[1] - 1, 0, -1)
         for d in [pts[:,ii] - pts[:,ii-1]]
         if np.sum(d**2) != 0))
    u /= np.sqrt(np.sum(u**2))
    v /= np.sqrt(np.sum(v**2))
    pts_ext = [(u*d + pts[:,0])[:,None], pts, (v*d + pts[:,-1])[:,None]]
    pts_ext = np.hstack(pts_ext)
    return pts_ext

def cross_isect_2D(segs1, segs2, rtol=1e-05, atol=1e-08):
    """Returns all intersections between the two line segment collections.

    `cross_isect(segs1, segs2)` returns a tuple `(ii1, ii2, xy)` of all the
    intersections of any line segment in `segs` with any line segment in
    `segs2`. The values `ii1` and `ii2` are the indices of the line segments
    where each intersection occurs, and the value `xy` is an `n x 2` matrix of
    the `n` intersection points.
    """
    from neuropythy.geometry.util import segment_intersection_2D
    (segs1, segs2) = (np.asarray(segs1), np.asarray(segs2))
    (n1, n2) = (segs1.shape[1] - 1, segs2.shape[1] - 1)
    ii1 = np.concatenate([[k]*n2 for k in range(n1)])
    ii2 = np.arange(n2)
    ii2 = np.concatenate([ii2]*n1)
    pts = segment_intersection_2D([segs1[...,ii1], segs1[...,ii1+1]],
                                  [segs2[...,ii2], segs2[...,ii2+1]],
                                  inclusive=True)
    pts = np.asarray(pts)
    ii = np.all(np.isfinite(pts), axis=0)
    (ii1, ii2, pts) = (ii1[ii], ii2[ii], pts[:,ii])
    # We have to be careful with sorting: if two of the ii1 segments are
    # the same, we want to order them by the distance from the segment
    # start.
    seglen2s = np.sum((segs1[:, :-1] - segs1[:, 1:])**2, axis=0)
    ds = [
        np.sum(seglen2s[:ii1[ii]]) + np.sum((segs1[:, ii1[ii]] - pts[:,ii])**2)
        for ii in range(len(ii1))]
    ii = np.argsort(ds)
    (ii1, ii2, pts) = (ii1[ii], ii2[ii], pts[:,ii])
    # See if we have accidentally created identical points.
    for ii in reversed(range(pts.shape[1] - 1)):
        if np.isclose(pts[:,ii], pts[:,ii+1], atol=atol, rtol=rtol).all():
            ii1 = np.delete(ii1, ii+1)
            ii2 = np.delete(ii2, ii+1)
            pts = np.delete(pts, ii+1, axis=-1)
    return (ii1, ii2, pts)

def iscloser(x, a, b):
    """Detects which of two points another point is closer to.

    `iscloser(x, a, b)` returns `True` if the point `x` is closer to the point
    `a` than it is to the point `b`. Otherwise, `False` is returned.
    """
    da2 = np.sum((x - a)**2)
    db2 = np.sum((x - b)**2)
    return (da2 < db2)

def fix_polygon(poly):
    """Given the points of a polygon, performs minor fixes and returns the fixed
    points.

    `fix_polygon(poly)` fixes duplicate adjacent points and checks for
    self-intersections.
    """
    # First, remove duplicate adjacent points.
    iidup = np.all(np.isclose(poly, np.roll(poly, -1, axis=1)), axis=0)
    iidup[-1] = False
    poly = poly[:, ~iidup]
    if not np.all(np.isclose(poly[:,0], poly[:,-1])):
        poly = np.hstack([poly, poly[:,[0]]])
    # If the polygon self-intersects, we really have two polygons and we just
    # want the largest of these.
    (ii1, ii2, pts) = cross_isect_2D(poly, poly)
    n = poly.shape[1]
    # Find where the intersections aren't with the neighboring segments.
    delta = np.mod(ii1 - ii2 + n//2, n-1) - n//2
    ii = np.where(np.abs(delta) > 1)[0]
    # If there are none of these, no problem!
    if len(ii) == 0:
        return poly
    elif len(ii) != 2:
        raise ValueError(f"{len(ii)} self-intersections in polygon")
    # We find which of the two polygons is the biggest (longest perimeter).
    # First split them into the two polygons.
    (ii1, ii2) = sorted(ii1[ii])
    pt = pts[:,ii[0]]
    poly1 = np.hstack([pt[:,None], poly[:, ii1+1:ii2], pt[:,None]])
    poly2 = np.hstack([pt[:,None], poly[:, ii2+1:], poly[:, :ii1], pt[:,None]])
    # Which has the larger perimeter?
    per1 = np.sum(np.sqrt(np.sum((poly1 - np.roll(poly1, -1, axis=1))**2, 0)))
    per2 = np.sum(np.sqrt(np.sum((poly2 - np.roll(poly2, -1, axis=1))**2, 0)))
    if per1 < per2:
        return poly2
    else:
        return poly1

def contour_endangle(pts, v):
    """Returns the angle between the last segment of a contour and another
    point.

    Given a `2 x n` contour matrix `contour` and a single point `x`, the
    function `contour_endangle(contour, x)` is the angle between the vector
    `contour[:,-1] - contour[:,-2]` and the vector `x - contour[:,-1]`. The
    cosine of the angle is returned.
    """
    u = pts[:, -1] - pts[:, -2]
    v = v - pts[:, -1]
    return u.dot(v)

def order_nearness(pts, startpt):
    """Order a contour matrix, according to closeness to a reference point.

    `order_nearness(contour, refpt)` returns `contour` if the first point in the
    `2 x n` contour matrix is closer to the `refpt` than the last point in the
    matrix; otherwise, it returns `fliplr(contour)`.
    """
    ends = pts[:,[0,-1]].T
    dists = [np.sqrt(np.sum((end - startpt)**2)) for end in ends]
    return np.fliplr(pts) if dists[1] < dists[0] else pts

def dedup_points(pts):
    """Returns a copy of the given contour with duplicate points removed."""
    iidup = np.all(np.isclose(pts, np.roll(pts, -1, axis=1)), axis=0)
    iidup[-1] = False
    return pts[:, ~iidup]

def seg_nearest(seg, point, argmin=False):
    """Returns the point on a segment that is nearest to a point.

    `seg_nearest(seg, point)` returns a `2 x N` matrix of each of the points
    nesrest the segments. The points argument must be a 2D vector, but the
    segments argument may contain a single segment or multiple; either way, the
    statement `((a_x, a_y), (b_x, b_y)) = seg` must extract the segments a and b
    whether `a_x`, `a_y`, `b_x`, and `b_y` are all vectors or numbers.

    If the optional argument `argmin` is set to an integer, then the nearest
    `argmin` points are returned along with the index of the segment containing
    them as a tuple `(nearest_points, segment_index)`.
    """
    (a, b) = seg = np.asarray(seg)
    point = np.asarray(point)
    if point.shape != a.shape:
        point = np.ones(a.shape) * point[:,None]
    ab = b - a
    seg_l2 = np.sum(ab**2, axis=0)
    ok = (seg_l2 > 0)
    inv_segl2 = np.ones_like(seg_l2, dtype=float)
    inv_segl2[ok] /= seg_l2[ok]
    t = np.sum(ab*(point - a), axis=0) * inv_segl2
    q = a + t*ab # The nearest point on the line ab to `point`
    qa_l2 = np.sum((a - q)**2, axis=0)
    qb_l2 = np.sum((b - q)**2, axis=0)
    q_onseg = np.isclose(np.sqrt(seg_l2), np.sqrt(qa_l2) + np.sqrt(qb_l2))
    pa_l2 = np.sum((a - point)**2, axis=0)
    pb_l2 = np.sum((b - point)**2, axis=0)
    ii = np.argmin([pa_l2, pb_l2], axis=0)
    ii *= ~q_onseg
    ii += q_onseg * 2
    if len(a.shape) == 1:
        return np.array([a, b, q])[ii, :]
    nears = np.array([a, b, q])[ii, :, np.arange(len(ii))].T
    if not argmin:
        return nears
    l2s = np.sum((nears - point)**2, axis=0)
    if argmin is True:
        ii = np.argmin(l2s)
        return (nears[:, ii], ii)
    ii = np.argsort(l2s)[:argmin]
    return (nears[:, ii], ii)

def find_crossings(crosser, outer):
    """Finds the crossing points of a crosser over a loop.

    `find_crossings(crosser, outer)` finds the two points at which the `crosser`
    contour intersects the `outer` contour. Both contours must be `2 x n` 
    matrices of points (the number `n` of points need not be the same for both
    contours, though). If more than 2 crossings are found, then the two crossing
    points that are adjacent and maximally distant from each other are returned.
    """
    (cii, oii, pts) = cross_isect_2D(crosser, outer)
    if len(cii) > 2:
        # We can probably cut this down to 2 by keeping the two
        # intersections that corner off the largest stretch of the crossing
        # contour.
        mx = np.argmax(np.diff(cii))
        ii = [mx, mx+1]
        (cii, oii, pts) = (cii[ii], oii[ii], pts[:, ii])
    return (cii, oii, pts)

# the a and b matrices are D x N (D: number of dimensions; N: number of observations)
def centroid(a, weights=None):
    """Returns the centroid of `a`.
    
    `centroid(a)` returns the mean of the rows of `a`.
    
    `centroid(a, w)` returns `dot(a, w) / sum(w)`.
    """
    if weights is None:
        return np.mean(a, axis=1)
    else:
        total_weight = np.sum(weights)
        return np.dot(a, weights) / total_weight
def centroid_align_points(a, b, weights=None, out=None):
    """Aligns the centroid of matrix `a` to that of `b`.
    
    `centroid_align_points(a, b)` aligns the points in the matrix `a` to those
    in the matrix `b` by aligning their centroids.
    
    Parameters
    ----------
    a : matrix
        The matrix that is to be aligned to matrix `b`. The shape of a can be
        any shape `(d,n)` where `d` is the number of dimensions, and `n` is the
        number of points.
    b : matrix
        The matrix to which `a` is to be aligned. Must be the same shape as `a`.
    weights : vector or None, optional
        The weights or masses to use in calculating the center of mass. The
        default is `None`.
    out : matrix or None, optional
        Where to store the result. If `None` (the default), then a new array is
        returned. Otherwise, the result is placed in `out`.

    Returns
    -------
    matrix
        The matrix `a` aligned to `b`. If `inplace` is `True` and `a` is a numpy
        array, `a` is returned.
    """
    centroid_a = centroid(a, weights)
    centroid_b = centroid(b, weights)
    if out is None:
        out = np.array(a)
    elif out is not a:
        out = np.asarray(out)
    else:
        out = np.asarray(out)
        out[...] = a
    out += (centroid_b - centroid_a)[:,None]
    return out
def rotation_alignment_matrix(a, b, weights=None):
    """Returns the rotation matrix that, when applied to `a`, aligns `a` to `b`.
    
    `rotation_alignment_matrix(a, b)` returns the rotation matrix that aligns
    `a` to `b`; i.e. if `r = rotation_alignment_matrix(a, b)`, then `dot(r, a)`
    minimizes the difference between `a` and `b`.
    
    `rotation_alignment_matrix(a, b, w)` uses `w` as a weight matrix such that
    the return value minimizes the weighted difference between `a` and `b`.
    """
    # First calculate the covariance matrix.
    if weights is None:
        cov = np.dot(b, np.transpose(a))
    else:
        weights = np.asarray(weights)
        cov = np.dot(b * weights[None,:], np.transpose(a))
        cov /= np.sum(weights)
    # Next, calculate the singular value decomposition.
    (u,s,vt) = np.linalg.svd(cov, compute_uv=True)
    det_u = np.linalg.det(u)
    det_v = np.linalg.det(vt)
    d = np.eye(len(a), dtype=np.asarray(a).dtype)
    d[-1,-1] = np.sign(det_v * det_u)
    return np.dot(np.dot(u, d), vt)
def rotation_align_points(a, b, weights=None, out=None):
    """Aligns the centroid of matrix `a` to that of `b` using rotation.
    
    `rotation_align_points(a, b)` aligns the points in the matrix `a` to those
    in the matrix `b` by rotating `a`.
    
    Parameters
    ----------
    a : matrix
        The matrix that is to be aligned to matrix `b`. The shape of a can be
        any shape `(d,n)` where `d` is the number of dimensions, and `n` is the
        number of points.
    b : matrix
        The matrix to which `a` is to be aligned. Must be the same shape as `a`.
    weights : vector or None, optional
        The weights or masses to use in calculating the center of mass. The
        default is `None`.
    out : matrix or None, optional
        Where to store the result. If `None` (the default), then a new array is
        returned. Otherwise, the result is placed in `out`.

    Returns
    -------
    matrix
        The matrix `a` aligned to `b`. If `inplace` is `True` and `a` is a numpy
        array, `a` is returned.
    """
    rotation_matrix = rotation_alignment_matrix(a, b, weights=weights)
    if out is not None:
        out = np.ascontiguousarray(out)
    return np.dot(rotation_matrix.astype(out.dtype), a, out=out)
def rigid_align_points(a, b, weights=None, out=None):
    """Rigidly aligns the points in matrix `a` to the points in matrix `b`.
    
    `rigid_align_points(a, b)` aligns the points in the matrix `a` to those in
    the matrix `b` by first aligning the centroid of `a` to that of `b` then
    finding and applying the rotation matrix that minimizes the differences
    between `a` and `b` using the Kabsch-Umeyama algorithm.
    
    Parameters
    ----------
    a : matrix
        The matrix that is to be aligned to matrix `b`. The shape of a can be
        any shape `(d,n)` where `d` is the number of dimensions, and `n` is the
        number of points.
    b : matrix
        The matrix to which `a` is to be aligned. Must be the same shape as `a`.
    weights : vector or None, optional
        The weights or masses to use in calculating the center of mass and the
        covariance matrix. The default is `None`.
    out : matrix or None, optional
        Where to store the result. If `None` (the default), then a new array is
        returned. Otherwise, the result is placed in `out`.

    Returns
    -------
    matrix
        The matrix `a` aligned to `b`. If `inplace` is `True` and `a` is a numpy
        array, `a` is returned.
    """
    # First, find the centroids.
    centroid_a = centroid(a, weights=weights)[:,None]
    centroid_b = centroid(b, weights=weights)[:,None]
    if out is a:
        out = np.ascontiguousarray(a)
    elif out is None:
        out = np.array(a, order='C')
    else:
        out = np.ascontiguousarray(out)
        out[...] = a
    out -= centroid_a
    # Then do the rotation. Whether inplace was True or False, it is now okay to
    # write to a (because either inplace is True or because the 
    # centroid_align_points function returned a new matrix for us).
    out = rotation_align_points(out, b - centroid_b, weights=weights, out=out)
    # Recenter at b's centroid.
    out += centroid_b
    # That's it.
    return out
def rigid_align_cortices(hem1, hem2, surface='white'):
    """Rigidly aligns two cortical surfaces.
    
    `rigid_align_cortices(hem1, hem2, surf)` is roughly equivalent to the
    function rigid_align_points(s1.coordinates, s2.coordinates)` where `s1` and
    `s2` are `hem1.surface(surf)` and `hem2.surface(surf)` respectively. The
    return value is a surface.
    """
    # Get the two surfaces.
    surf1 = hem1.surface(surface)
    surf2 = hem2.surface(surface)
    # Conform surface1 to surface 2.
    surf1_as_surf2 = hem2.interpolate(hem1, surf2.coordinates)
    # Then align surface 1 with the conformed surface 1.
    surf1_on_surf2 = rigid_align_points(surf1.coordinates, surf1as2)
    # Return the surface with the new coordinates.
    return surf1.copy(coordinates=surf1on2)
