#!/usr/bin/env python3
#
# Landscape generator for The Sentinel (aka The Sentry)
#
# Generates the landscapes and placed object from the original game.
#
# By Simon Owen https://github.com/simonowen/sentland

import argparse
import sys
from enum import IntEnum
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

import numpy as np
import numpy.typing as npt

array2d = npt.NDArray[np.int8]

num_landscapes = 0xE000  # includes extended hex landscapes
ull = 0
rng_usage = 0


class ObjType(IntEnum):
    NONE = -1
    ROBOT = 0
    SENTRY = 1
    TREE = 2
    BOULDER = 3
    MEANIE = 4
    SENTINEL = 5
    PEDESTAL = 6


class Object:
    def __init__(self, type: ObjType, x: int, y: int, z: int) -> None:
        self.type = type
        self.x = x
        self.y = y
        self.z = z
        self.rot: int | None = None
        self.step: int | None = None
        self.timer: int | None = None

    def __str__(self) -> str:
        """Generate string representation of object"""
        name = self.type.name.capitalize()
        rotdeg = None if self.rot is None else f"{int(self.rot * 360 / 256):03}\u00b0"
        rotdir = "" if self.step is None else " \u21ba" if self.step < 0 else " \u21bb"

        s = f"{name+':':9s} x={self.x:02X} y={self.y:02X} z={self.z:02X}"
        if self.rot is not None:
            s += f" rot={self.rot:02X} ({rotdeg}{rotdir})"
        if self.timer is not None:
            s += f" next={self.timer:02X}"
        return s


def get_offset(x: int, z: int) -> int:
    """Convert x and z coordinates to linear offset into game map data"""
    return ((x & 3) << 8) | ((x & 0x1C) << 3) | z


def get_x_z(offset: int) -> tuple[int, int]:
    """Convert linear game map data offset to x and z coordinates"""
    x = ((offset & 0x300) >> 8) | ((offset & 0xE0) >> 3)
    z = offset & 0x1F
    return x, z


def at_offset(maparr: array2d, offset: int) -> int:
    """Return map entry at given original game map offset"""
    x, z = get_x_z(offset)
    return int(maparr[z][x])


def shape_at(x: int, z: int, maparr: array2d) -> int:
    """Return map code at given map location"""
    return int(maparr[z][x]) & 0xF


def height_at(x: int, z: int, maparr: array2d) -> int:
    """Return map height at given location"""
    return int(maparr[z][x]) >> 4


def is_flat(x: int, z: int, maparr: array2d) -> bool:
    """Return True if the map location is a flat tile"""
    return shape_at(x, z, maparr) == 0


def objects_at(x: int, z: int, objects: list[Object]) -> list[Object]:
    """Return a list of objects stacked at map location"""
    return [o for o in objects if o.x == x and o.z == z]


def wrapped_slice(maparr: array2d, entries: int = 0x23, *, x: int | None = None, z: int | None = None) -> list[int]:
    """Return x or z slice from the map, wrapped around at the edges"""
    if z is not None:
        return [maparr[z][x & 0x1F] for x in range(entries)]
    else:
        return [maparr[z & 0x1F][x] for z in range(entries)]


def smooth_slice(arr: list[int]) -> list[int]:
    """Smooth a map slice by averaging neighbouring groups of values"""
    group_size = len(arr) - 0x1F
    return [
        sum(arr[x:(x + group_size)]) // group_size
        for x in range(len(arr) - group_size + 1)
    ]


def smooth_map(maparr: array2d, axis: str) -> array2d:
    """Smooth the map by averaging groups across the given axis"""
    new_maparr = np.empty_like(maparr)

    for i in range(0x20):
        if axis == "x":
            new_maparr[:, i] = smooth_slice(wrapped_slice(maparr, x=i))
        else:
            new_maparr[i, :] = smooth_slice(wrapped_slice(maparr, z=i))

    return new_maparr


def despike_midval(arr: list[int]) -> int:
    """Smooth 3 map vertices, returning a new central vertex height"""
    if arr[1] == arr[2]:
        return arr[1]
    elif arr[1] > arr[2]:
        if arr[1] <= arr[0]:
            return arr[1]
        elif arr[0] < arr[2]:
            return arr[2]
        else:
            return arr[0]
    elif arr[1] >= arr[0]:
        return arr[1]
    elif arr[2] < arr[0]:
        return arr[2]
    else:
        return arr[0]


def despike_slice(arr: list[int]) -> list[int]:
    """Smooth a slice by flattening single vertex peaks and troughs"""
    arr_copy = arr[:]
    for x in reversed(range(0x20)):
        arr_copy[x + 1] = despike_midval(arr_copy[x:(x + 3)])
    return arr_copy[:32]


def despike_map(maparr: array2d, axis: str) -> array2d:
    """De-spike the map in slices across the given axis"""
    new_map = np.empty_like(maparr)

    for i in range(0x20):
        if axis == "x":
            new_map[:, i] = despike_slice(wrapped_slice(maparr, x=i))
        else:
            new_map[i, :] = despike_slice(wrapped_slice(maparr, z=i))

    return new_map


def scale_and_offset(val: int, scale: int = 0x18) -> int:
    """Scale and offset values to generate vertex heights"""
    mag = val - 0x80  # 7-bit signed range
    mag = mag * scale // 256  # scale and use upper 8 bits
    mag = max(mag + 6, 0)  # centre at 6 and limit minimum
    mag = min(mag + 1, 11)  # raise by 1 and limit maximum
    return mag


def tile_shape(fl: int, bl: int, br: int, fr: int) -> int:
    """Determine tile shape code from 4 vertex heights"""
    if fl == fr:
        if fl == bl:
            if fl == br:
                shape = 0
            elif fl < br:
                shape = 0xA
            else:
                shape = 0x3
        elif br == bl:
            if br < fr:
                shape = 0x1
            else:
                shape = 0x9
        elif br == fr:
            if br < bl:
                shape = 0x6
            else:
                shape = 0xF
        else:
            shape = 0xC
    elif fl == bl:
        if br == fr:
            if br < bl:
                shape = 0x5
            else:
                shape = 0xD
        elif br == bl:
            if br < fr:
                shape = 0xE
            else:
                shape = 0x7
        else:
            shape = 0x4
    elif br == fr:
        if br == bl:
            if br < fl:
                shape = 0xB
            else:
                shape = 0x2
        else:
            shape = 0x4
    else:
        shape = 0xC

    return shape


def add_tile_shapes(maparr: array2d) -> array2d:
    """Add tile shape code to upper 4 bits of each tile"""
    new_maparr = np.copy(maparr)
    for z in reversed(range(0x1F)):
        for x in reversed(range(0x1F)):
            fl = maparr[z + 0, x + 0] & 0xF
            bl = maparr[z + 1, x + 0] & 0xF
            br = maparr[z + 1, x + 1] & 0xF
            fr = maparr[z + 0, x + 1] & 0xF
            shape = tile_shape(fl, bl, br, fr)
            new_maparr[z][x] = (shape << 4) | (maparr[z][x] & 0xF)
    return new_maparr


def swap_nibbles(maparr: array2d) -> array2d:
    """Swap upper and lower 4 bits in each map byte"""
    return np.array([[((maparr[z][x] & 0xF) << 4) | (maparr[z][x] >> 4)
                    for x in range(0x20)] for z in range(0x20)])


def seed(landscape_bcd: int) -> None:
    """Seed RNG using landscape number"""
    global ull, rng_usage
    ull = (1 << 16) | landscape_bcd
    rng_usage = 0


def rng() -> int:
    """Pull next 8-bit value from random number generator"""
    global ull, rng_usage
    for _ in range(8):
        ull <<= 1
        ull |= ((ull >> 20) ^ (ull >> 33)) & 1

    rng_usage += 1
    return (ull >> 32) & 0xFF


def rng_00_16() -> int:
    """Random number in range 0 to 0x16"""
    r = rng()
    return (r & 7) + ((r >> 3) & 0xF)


def arr_to_memory(maparr: array2d) -> bytes:
    """Convert array data to in-memory format used by game"""
    return bytes([at_offset(maparr, x) for x in range(1024)])


def verify(maparr: array2d, landscape_bcd: int, name: str) -> None:
    """Verify the map data against golden images, if they exist"""
    path = files("golden") / f"{landscape_bcd:04X}_{name}.bin"
    if path.is_file():
        with path.open("rb") as f:
            if f.read() != arr_to_memory(maparr):
                sys.exit(f"Data mismatch against {path}")


def generate_landscape(landscape_bcd: int, landscape_step: int) -> tuple[array2d, int]:
    """Generate landscape data for given landscape number"""
    # Seed RNG using landscape number in BCD.
    seed(landscape_bcd)

    # Read 81 values to warm the RNG.
    [rng() for _ in range(0x51)]

    # Random height scaling (but fixed value for landscape 0000!).
    height_scale = (rng_00_16() + 0x0E) if landscape_bcd else 0x18

    # Fill the map with random values (z from back to front, x from right to left).
    maparr = np.array(list(reversed([list(reversed([rng()
                      for x in range(0x20)])) for z in range(0x20)])))
    verify(maparr, landscape_bcd, "random")

    if landscape_step >= 2:
        # 2 passes of smoothing, each across z-axis then x-axis (pass 1).
        maparr = smooth_map(maparr, "z")
        maparr = smooth_map(maparr, "x")

    if landscape_step >= 3:
        # 2 passes of smoothing, each across z-axis then x-axis (pass 2).
        maparr = smooth_map(maparr, "z")
        maparr = smooth_map(maparr, "x")
        verify(maparr, landscape_bcd, "smooth3")

    if landscape_step >= 4:
        # Scale and offset values to give vertex heights in range 1 to 11.
        maparr = np.array([[scale_and_offset(int(x), height_scale) for x in z] for z in maparr])
        verify(maparr, landscape_bcd, "scaled")

    if landscape_step >= 5:
        # Two de-spike passes, each across z-axis then x-axis (pass 1).
        maparr = despike_map(maparr, "z")
        maparr = despike_map(maparr, "x")

    if landscape_step >= 6:
        # Two de-spike passes, each across z-axis then x-axis (pass 2).
        maparr = despike_map(maparr, "z")
        maparr = despike_map(maparr, "x")
        verify(maparr, landscape_bcd, "despike3")

    if landscape_step >= 4:
        # Add shape codes for each tile, to simplify examining the landscape.
        maparr = add_tile_shapes(maparr)
        if landscape_step >= 6:
            verify(maparr, landscape_bcd, "shape")

    if landscape_step >= 4:
        # Finally, swap the high and low nibbles in each byte for the final format.
        maparr = swap_nibbles(maparr)
        if landscape_step >= 6:
            verify(maparr, landscape_bcd, "swap")

    return maparr, height_scale


def view_landscape(maparr: array2d, landscape_bcd: int, num_sentries: int, landscape_step: int, export_file: str, view_landscape: bool, dark: bool, colour_objects: bool, objects: list[Object]) -> None:
    """Crude viewing of generated landscape data"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LinearLocator
        from matplotlib.transforms import Bbox
    except ModuleNotFoundError:
        sys.exit("Landscape viewing requires matplotlib package")

    axis = np.arange(0, 0x20, 1)
    X, Y = np.meshgrid(axis, axis)
    if landscape_step >= 4:
        Z = np.array(maparr) >> 4  # map just height nibble
    else:
        Z = np.array(maparr)  # map full height range of 255

    flat_colours1 = (
        (0.0, 1.0, 0.0), (1.0, 1.0, 0.62), (0.62, 1.0, 1.0), (1.0, 1.0, 0.62),
        (1.0, 1.0, 1.0), (1.0, 0.75, 0.75), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0))
    flat_colours2 = (
        (0.0, 0.62, 0.62), (0.62, 0.0, 0.62), (0.0, 0.62, 0.62), (0.87, 0.37, 0.0),
        (0.37, 0.37, 1.0), (1.0, 0.0, 0.0), (0.62, 0.0, 0.62), (0.37, 0.37, 1.0))


    flat_colours = (flat_colours1[num_sentries], flat_colours2[num_sentries])
    slope_colours = ((0.6, 0.6, 0.6), (0.7, 0.7, 0.7))  # light grey, dark grey

    # Sentinel is red or blue
    sentinel_colours = (
        (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    # Sentries are orange or light blue
    sentry_colours = (
        (1.0, 0.5, 0.3), (0.0, 0.5, 1.0), (1.0, 0.5, 0.3), (0.0, 0.5, 1.0),
        (1.0, 0.5, 0.3), (0.0, 0.5, 1.0), (1.0, 0.5, 0.3), (1.0, 0.5, 0.3))

    # Player is yellow or magenta
    player_colours = (
        (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0))

    # Trees are magenta or green
    tree_colours = (
        (1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))

    colors = np.empty(X.shape, dtype="3f")
    for y in range(len(Y)):
        for x in range(len(X)):
            if maparr[y][x] & 0xF:
                colors[y, x] = slope_colours[(x + y) & 1]
            else:
                if colour_objects:
                    object_stack = objects_at(x, y, objects)
                    if object_stack:
                        if object_stack[0].type == ObjType.SENTINEL or object_stack[0].type == ObjType.PEDESTAL:
                            colors[y, x] = sentinel_colours[num_sentries]
                        elif object_stack[0].type == ObjType.SENTRY:
                            colors[y, x] = sentry_colours[num_sentries]
                        elif object_stack[0].type == ObjType.ROBOT:
                            colors[y, x] = player_colours[num_sentries]
                        elif object_stack[0].type == ObjType.TREE:
                            colors[y, x] = tree_colours[num_sentries]
                    else:
                        colors[y, x] = flat_colours[(x + y) & 1]
                else:
                    colors[y, x] = flat_colours[(x + y) & 1]

    if dark:
        plt.style.use("dark_background")
        plt.rcParams['grid.color'] = (0.4, 0.4, 0.4, 1.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.xaxis.set_ticks([0, 10, 20, 30])
    ax.yaxis.set_ticks([0, 10, 20, 30])

    if dark:
        ax.xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        ax.yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        ax.zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))

    if landscape_step >= 7:
        ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)
    else:
        ax.scatter(X, Y, Z, s=2)

    if landscape_step >= 4:
        ax.set_zlim(1, 11)
        ax.zaxis.set_major_locator(LinearLocator(6))
    else:
        ax.set_zlim(0, 255)
        ax.zaxis.set_ticks([0, 64, 128, 192, 255])

    match landscape_step:
        case 1:
            plt.title(f"Landscape {landscape_bcd:04X}\nSteps 1-4: Seed tile data")
        case 2:
            plt.title(f"Landscape {landscape_bcd:04X}\nStep 5: Smooth by average (pass 1)")
        case 3:
            plt.title(f"Landscape {landscape_bcd:04X}\nStep 6: Smooth by average (pass 2)")
        case 4:
            plt.title(f"Landscape {landscape_bcd:04X}\nStep 7: Scale and cap")
        case 5:
            plt.title(f"Landscape {landscape_bcd:04X}\nStep 8: Smooth spikes (pass 1)")
        case 6:
            plt.title(f"Landscape {landscape_bcd:04X}\nStep 9: Smooth spikes (pass 2)")
        case 7:
            plt.title(f"Landscape {landscape_bcd:04X}\nSteps 10-11: Calculate tile shapes")
        case _:
            plt.title(f"Landscape {landscape_bcd:04X}")

    if export_file:
        plt.savefig(export_file, dpi=144, bbox_inches=Bbox([[1.0, 0.0], [5.7, 4.81]]))
    if view_landscape:
        plt.show()


def calc_num_sentries(landscape_bcd: int) -> int:
    """Determine number of sentries on landscape"""
    # Only ever the Sentinel on the first landscape.
    if landscape_bcd == 0x0000:
        return 1

    # Base count uses landscape BCD thousands digit, offset by 2.
    base_sentries = ((landscape_bcd & 0xF000) >> 12) + 2

    while True:
        r = rng()
        # count leading zeros on b6-0 for adjustment size
        adjust = (format(r & 0x7F, "07b") + "1").find("1")
        # b7 determines adjustment sign (note: 1s complement)
        if r & 0x80:
            adjust = ~adjust

        num_sentries = base_sentries + adjust
        if 0 <= num_sentries <= 7:
            break

    # Levels under 100 use tens digit to limit number of sentries.
    max_sentries = (landscape_bcd & 0x00F0) >> 4
    if landscape_bcd >= 0x0100 or max_sentries > 7:
        max_sentries = 7

    # Include Sentinel in sentry count.
    return 1 + min(num_sentries, max_sentries)


def highest_positions(maparr: array2d) -> list[list[int]]:
    """Find the highest placement positions in 4x4 regions on the map"""
    grid_max = []

    # Scan the map as 64 regions of 4x4 (less one on right/back edges)
    # in z order from front to back and x from left to right.
    for i in range(0x40):
        gridx, gridz = ((i & 7) << 2), ((i & 0x38) >> 1)
        max_height, max_x, max_z = 0, -1, -1

        # Scan each 4x4 region, z from front to back, x from left to right.
        for j in range(0x10):
            x, z = gridx + (j & 3), gridz + (j >> 2)

            # The back and right edges are missing a tile, so skip.
            if x == 0x1F or z == 0x1F:
                continue

            height = height_at(x, z, maparr)
            if is_flat(x, z, maparr) and height >= max_height:
                max_height, max_x, max_z = height, x, z

        grid_max.append([max_height, max_x, max_z])

    return grid_max


def object_at(type: ObjType, x: int, y: int, z: int) -> Object:
    """Place object at given position but with random rotation"""
    obj = Object(type, x, y, z)

    # Random rotation, limited to 32 steps, biased by +135 degrees.
    obj.rot = ((rng() & 0xF8) + 0x60) & 0xFF
    return obj


def random_coord() -> int:
    """Calculate random map axis coordinate"""
    while True:
        r = rng() & 0x1F
        if r < 0x1F:
            return r


def object_random(type: ObjType, max_height: int, objects: list[Object], maparr: array2d) -> Object:
    """Generate given object at a random unused position below the given height"""
    while True:
        for attempt in range(0xFF):
            x, z = random_coord(), random_coord()
            y = height_at(x, z, maparr)

            if (
                is_flat(x, z, maparr)
                and not objects_at(x, z, objects)
                and y < max_height
            ):
                return object_at(type, x, y, z)

        max_height += 1
        if max_height >= 0xC:
            raise RuntimeError(f"Unable to place {type.name} on landscape")


def place_sentries(landscape_bcd: int, maparr: array2d) -> tuple[list[Object], int]:
    """Place Sentinel and appropriate sentry count for given landscape"""
    objects: list[Object] = []
    highest = highest_positions(maparr)
    max_height = max([x[0] for x in highest])

    num_sentries = calc_num_sentries(landscape_bcd)
    for _ in range(num_sentries):
        while True:
            # Filter for high positions at the current height limit.
            height_indices = [i for i, x in enumerate(highest) if x[0] == max_height]
            if height_indices:
                break

            # No locations so try 1 level down, stopping at zero.
            max_height -= 1
            if max_height == 0:
                return objects, max_height

        # Results are in reverse order due to backwards 6502 iteration loop.
        height_indices = list(reversed(height_indices))

        # Mask above number of entries to limit random scope.
        idx_mask = 0xFF >> format(len(height_indices), "08b").find("1")
        while True:
            idx = rng() & idx_mask
            if idx < len(height_indices):
                break

        idx_grid = height_indices[idx]
        y, x, z = highest[idx_grid]

        # Invalidate the selected and surrounding locations by setting zero height.
        for offset in [-9, -8, -7, -1, 0, 1, 7, 8, 9]:
            idx_clear = idx_grid + offset
            if idx_clear >= 0 and idx_clear < len(highest):
                highest[idx_clear][0] = 0

        if not objects:
            pedestal = object_at(ObjType.PEDESTAL, x, y, z)
            pedestal.rot = 0
            objects.append(pedestal)
            objects.append(object_at(ObjType.SENTINEL, x, y + 1, z))
        else:
            objects.append(object_at(ObjType.SENTRY, x, y, z))

        # Generate rotation step/direction and timer delay from RNG.
        r = rng()
        objects[-1].step = -20 if (r & 1) else +20
        objects[-1].timer = ((r >> 1) & 0x1F) | 5

    return objects, max_height


def place_player(landscape_bcd: int, max_height: int, objects: list[Object], maparr: array2d) -> tuple[list[Object], int]:
    """Place player robot on the landscape"""

    # The player position is fixed on landscape 0000.
    if landscape_bcd == 0x0000:
        x, z = 0x08, 0x11
        player = object_at(ObjType.ROBOT, x, height_at(x, z, maparr), z)
    else:
        # Player is never placed above height 6.
        max_player_height = min(max_height, 6)
        player = object_random(ObjType.ROBOT, max_player_height, objects, maparr)

    objects.append(player)
    return objects, max_height


def place_trees(max_height: int, objects: list[Object], maparr: array2d) -> tuple[list[Object], int]:
    """Place the appropriate number of trees for the sentry count"""

    # Count the placed Sentinel and sentries.
    num_sents = len(
        [o for o in objects if o.type in [ObjType.SENTINEL, ObjType.SENTRY]]
    )

    r = rng()
    max_trees = 48 - (3 * num_sents)
    num_trees = (r & 7) + ((r >> 3) & 0xF) + 10
    num_trees = min(num_trees, max_trees)

    for _ in range(num_trees):
        tree = object_random(ObjType.TREE, max_height, objects, maparr)
        objects.append(tree)

    return objects, max_height


def generate_level(landscape_bcd: int, landscape_step: int) -> tuple[array2d, list[Object], int]:
    """Generate landscape level data and placed objects"""
    maparr, height_scale = generate_landscape(landscape_bcd, landscape_step)

    objects, max_height = place_sentries(landscape_bcd, maparr)
    objects, max_height = place_player(landscape_bcd, max_height, objects, maparr)
    if landscape_step >= 6:
        objects, max_height = place_trees(max_height, objects, maparr)

    return maparr, objects, height_scale


def args_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    try:
        pkg_version = version('sentland')
    except PackageNotFoundError:
        pkg_version = 'unknown'

    parser = argparse.ArgumentParser(
        description="Landscape generator for The Sentinel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("landscape",
                        help="landscape number", type=lambda x: int(x, 16), nargs="?")
    parser.add_argument("-v", "--view",
                        help="view landscape in matplot", action="store_true", default=False)
    parser.add_argument("-o", "--output",
                        help="output file name", type=str, default=None)
    parser.add_argument("-m", "--memory",
                        help="save data in game memory format", action="store_true", default=False)
    parser.add_argument("-q", "--quiet",
                        help="suppress output messages", action="store_true", default=False)
    parser.add_argument('-V', '--version',
                        action='version', version=f'%(prog)s {pkg_version}')
    parser.add_argument("-s", "--step",
                        help="stop generating landscape at step 1-7", type=int, default=8)
    parser.add_argument("-t", "--tileinfo",
                        help="output data about tiles and shapes", action="store_true", default=False)
    parser.add_argument("-x", "--xdata",
                        help="output extra data about the landscape", action="store_true", default=False)
    parser.add_argument("-e", "--export",
                        help="export image as a png (disables -v)", default=None)
    parser.add_argument("-d", "--dark",
                        help="view landscape in the dark", action="store_true", default=False)
    parser.add_argument("-c", "--colourobjects",
                        help="colour tiles containing objects", action="store_true", default=False)
    return parser


def main() -> None:
    """Main entry point for command line execution"""
    parser = args_parser()
    args = parser.parse_args()

    if args.landscape is None:
        parser.print_help()
    elif args.landscape < 0 or args.landscape >= num_landscapes:
        sys.exit(f"Landscape number must be in range 0000-{num_landscapes-1:04X}")
    else:
        maparr, objects, height_scale = generate_level(args.landscape, args.step)

        if args.output:
            with open(args.output, "wb") as f:
                if args.memory:
                    f.write(arr_to_memory(maparr))
                else:
                    f.write(np.array(maparr, dtype="B").tobytes())
            if not args.quiet:
                print(f"Wrote landscape {args.landscape:04X} to {args.output}")

        if not args.quiet:
            if args.xdata:
                # Landscape - Multiplier - Tree count - Sentry count
                trees = 0
                sentries = 0
                for o in objects:
                    if o.type == ObjType.TREE:
                        trees += 1
                    if o.type == ObjType.SENTRY:
                        sentries += 1
                print("{:04X}\t{}\t{}\t{}".format(args.landscape, height_scale, trees, sentries))
            else:
                print("Landscape: {:04d}\n".format(args.landscape))

            if args.tileinfo:
                print("Tile data multiplier (14-36): {}\n".format(height_scale))

                print("Objects:")
                for o in objects:
                    print(f"  {o}")
                print()

                print("Shapes:\n")
                print("y x ", end='')
                for x in range(0, 31):
                    print("{:>2} ".format(x), end='')
                print()
                for y in range(30, -1, -1):
                    print("{:>2} ".format(y), end='')
                    for x in range(0, 31):
                        print("{:>3}".format(shape_at(x, y, maparr)), end='')
                    print()
                print()

                print("Altitudes:\n")
                print("y x ", end='')
                for x in range(0, 32):
                    print("{:>2} ".format(x), end='')
                print()
                for y in range(31, -1, -1):
                    print("{:>2} ".format(y), end='')
                    for x in range(0, 32):
                        print("{:>3}".format(height_at(x, y, maparr)), end='')
                    print()
                print()

                print("4a:\n")
                for x in range(0, 31):
                    for y in range(0, 31):
                        if shape_at(x, y, maparr) == 4:
                            s = height_at(x, y, maparr)
                            t = height_at(x, y + 1, maparr)
                            u = height_at(x + 1, y + 1, maparr)
                            v = height_at(x + 1, y, maparr)
                            if u == v:
                                min_height = min(s, t, u, v)
                                print("{:>2},{:>2}: {} {}  {} {}".format(x, y, t, u, t - min_height, u - min_height))
                                print("       {} {}  {} {}\n".format(s, v, s - min_height, v - min_height))

                print("4b:\n")
                for x in range(0, 31):
                    for y in range(0, 31):
                        if shape_at(x, y, maparr) == 4:
                            s = height_at(x, y, maparr)
                            t = height_at(x, y + 1, maparr)
                            u = height_at(x + 1, y + 1, maparr)
                            v = height_at(x + 1, y, maparr)
                            if s == t:
                                min_height = min(s, t, u, v)
                                print("{:>2},{:>2}: {} {}  {} {}".format(x, y, t, u, t - min_height, u - min_height))
                                print("       {} {}  {} {}\n".format(s, v, s - min_height, v - min_height))

                print("12a:\n")
                for x in range(0, 31):
                    for y in range(0, 31):
                        if shape_at(x, y, maparr) == 12:
                            s = height_at(x, y, maparr)
                            t = height_at(x, y + 1, maparr)
                            u = height_at(x + 1, y + 1, maparr)
                            v = height_at(x + 1, y, maparr)
                            if s != v:
                                min_height = min(s, t, u, v)
                                print("{:>2},{:>2}: {} {}  {} {}".format(x, y, t, u, t - min_height, u - min_height))
                                print("       {} {}  {} {}\n".format(s, v, s - min_height, v - min_height))

                print("12b:\n")
                for x in range(0, 31):
                    for y in range(0, 31):
                        if shape_at(x, y, maparr) == 12:
                            s = height_at(x, y, maparr)
                            t = height_at(x, y + 1, maparr)
                            u = height_at(x + 1, y + 1, maparr)
                            v = height_at(x + 1, y, maparr)
                            if s == v:
                                min_height = min(s, t, u, v)
                                print("{:>2},{:>2}: {} {}  {} {}".format(x, y, t, u, t - min_height, u - min_height))
                                print("       {} {}  {} {}\n".format(s, v, s - min_height, v - min_height))

        if args.view or args.export:
            num_sentries = len([o for o in objects if o.type == ObjType.SENTRY])
            view_landscape(maparr, args.landscape, num_sentries, args.step, args.export, args.view, args.dark, args.colourobjects, objects)


if __name__ == "__main__":
    main()
