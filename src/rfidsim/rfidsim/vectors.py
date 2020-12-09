import numpy as np
import collections
import decimal


def length(v):
    v = vec3(v)
    return np.sqrt(v.dot(v))


def normalize(v, tol=1e-9):
    v = vec3(v)
    if is_zero(v, tol=tol):
        raise ValueError("can not normalize vector v={}".format(v))
    return v / length(v)


def get_in_axes(v, x_axis=None, y_axis=None, z_axis=None):
    v = vec3(v)
    x_axis = vec3(x_axis) if x_axis is not None else None
    y_axis = vec3(y_axis) if y_axis is not None else None
    z_axis = vec3(z_axis) if z_axis is not None else None
    if (x_axis is None and
            (y_axis is None or z_axis is None) or
            (y_axis is None and z_axis is None)):
        raise ValueError("at least two axis must be non-zero: "
                         "x_axis={x}, y_axis={y}, z_axis={z}".format(
            x=x_axis, y=y_axis, z=z_axis))
    ox = x_axis if x_axis is not None else np.cross(y_axis, z_axis)
    oy = y_axis if y_axis is not None else np.cross(z_axis, x_axis)
    oz = z_axis if z_axis is not None else np.cross(x_axis, y_axis)
    x, y, z = v
    return x*ox + y*oy + z*oz


def get_angle(u, v, tol=1e-9):
    v = vec3(v)
    u = vec3(u)
    v_len = np.sqrt(v.dot(v))
    u_len = np.sqrt(u.dot(u))
    if v_len < tol or u_len < tol:
        raise ValueError("angle with zero vector undefined, v={v}, u={u}"
                         "".format(v=v, u=u))
    return np.arccos(v.dot(u) / (v_len * u_len))


def deg2rad(value, reduce=True):
    if reduce:
        value %= 360.
    return (value / 180.) * np.pi


def rad2deg(value, reduce=True):
    if reduce:
        value %= 2*np.pi
    return (value / np.pi) * 180.


def vec3(v):
    if not isinstance(v, collections.Iterable) or len(v) != 3:
        raise ValueError("failed to interpret v={} (type:{}) as 3D vector"
                         "".format(v, type(v)))
    return v if isinstance(v, np.ndarray) else np.array(v)


def is_zero(v, tol=1e-9):
    try:
        v = vec3(v)
        return v.dot(v) < tol*tol
    except ValueError:
        return decimal.Decimal(v) < tol
