"""Helpers previos para auto-seleccion por mouse (Simple Diamond)."""

import bmesh
from mathutils import Vector, geometry
from bpy_extras import view3d_utils

AUTO_AXIS_DOT = 0.75
AUTO_AXIS_CLUSTER_DOT = 0.9
AUTO_SELECT_RAILS = 4
AUTO_SELECT_LENGTH = 5
MOUSE_CACHE_KEY = "sdm_last_mouse"


def _find_view3d_region(context, mouse_x, mouse_y):
    """Resolver region y rv3d de un VIEW_3D cercano al mouse."""
    screen = context.screen
    if screen is None and context.window:
        screen = context.window.screen
    if screen is None:
        return None, None

    fallback = None
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        rv3d = area.spaces.active.region_3d if area.spaces.active else None
        if region is None or rv3d is None:
            continue
        if fallback is None:
            fallback = (region, rv3d)
        if mouse_x is None or mouse_y is None:
            continue
        if (
            mouse_x >= region.x
            and mouse_x <= region.x + region.width
            and mouse_y >= region.y
            and mouse_y <= region.y + region.height
        ):
            return region, rv3d
    return fallback if fallback is not None else (None, None)


def _collect_edge_dirs(
    seed: bmesh.types.BMVert,
    max_hops: int = 2,
) -> list[Vector]:
    """Recolectar direcciones de edges cercanos al vert activo."""
    dirs: list[Vector] = []
    visited = {seed}
    frontier = [seed]
    for _ in range(max_hops):
        next_frontier = []
        for v in frontier:
            for e in v.link_edges:
                other = e.other_vert(v)
                if not other.is_valid:
                    continue
                vec = other.co - v.co
                if vec.length < 1e-8:
                    continue
                dirs.append(vec.normalized())
                if other not in visited:
                    visited.add(other)
                    next_frontier.append(other)
        frontier = next_frontier
    return dirs


def _cluster_directions(
    dirs: list[Vector],
    dot_threshold: float,
) -> list[Vector]:
    """Agrupar direcciones por similitud y devolver ejes principales."""
    groups: list[list[Vector]] = []
    for d in dirs:
        placed = False
        for g in groups:
            if abs(d.dot(g[0])) >= dot_threshold:
                g.append(d)
                placed = True
                break
        if not placed:
            groups.append([d])

    axes: list[tuple[Vector, int]] = []
    for g in groups:
        ref = g[0]
        acc = Vector((0.0, 0.0, 0.0))
        for d in g:
            acc += d if d.dot(ref) >= 0.0 else -d
        if acc.length < 1e-8:
            continue
        acc.normalize()
        axes.append((acc, len(g)))
    axes.sort(key=lambda item: item[1], reverse=True)
    return [axis for axis, _ in axes]


def _pick_axes(active: bmesh.types.BMVert) -> list[Vector]:
    """Elegir dos ejes principales en la vecindad del vert activo."""
    dirs = _collect_edge_dirs(active, max_hops=2)
    if not dirs:
        return []
    candidates = _cluster_directions(dirs, AUTO_AXIS_CLUSTER_DOT)
    axes: list[Vector] = []
    for axis in candidates:
        if all(abs(axis.dot(a)) < 0.8 for a in axes):
            axes.append(axis)
        if len(axes) >= 2:
            break
    return axes


def _step_along_axis(
    vert: bmesh.types.BMVert,
    axis: Vector,
    min_dot: float,
) -> bmesh.types.BMVert | None:
    """Moverse al vecino mas alineado con el eje."""
    axis = axis.normalized()
    best = None
    best_dot = min_dot
    for e in vert.link_edges:
        other = e.other_vert(vert)
        if not other.is_valid:
            continue
        vec = other.co - vert.co
        if vec.length < 1e-8:
            continue
        vec.normalize()
        dot = vec.dot(axis)
        if dot > best_dot:
            best_dot = dot
            best = other
    return best


def _build_centered_rail(
    active: bmesh.types.BMVert,
    axis: Vector,
    count: int,
) -> list[bmesh.types.BMVert] | None:
    """Construir un rail centrado en el vert activo."""
    half = count // 2
    back = []
    cur = active
    for _ in range(half):
        nxt = _step_along_axis(cur, -axis, AUTO_AXIS_DOT)
        if nxt is None:
            return None
        back.append(nxt)
        cur = nxt
    fwd = []
    cur = active
    for _ in range(half):
        nxt = _step_along_axis(cur, axis, AUTO_AXIS_DOT)
        if nxt is None:
            return None
        fwd.append(nxt)
        cur = nxt
    rail = list(reversed(back)) + [active] + fwd
    if len(rail) != count or len(set(rail)) != count:
        return None
    return rail


def _build_centers(
    active: bmesh.types.BMVert,
    cross_axis: Vector,
    offsets: list[int],
) -> dict[int, bmesh.types.BMVert] | None:
    """Construir centros moviendose por el eje transversal."""
    centers = {0: active}
    max_pos = max(offsets)
    min_neg = min(offsets)
    cur = active
    for i in range(1, max_pos + 1):
        nxt = _step_along_axis(cur, cross_axis, AUTO_AXIS_DOT)
        if nxt is None:
            return None
        centers[i] = nxt
        cur = nxt
    cur = active
    for i in range(1, abs(min_neg) + 1):
        nxt = _step_along_axis(cur, -cross_axis, AUTO_AXIS_DOT)
        if nxt is None:
            return None
        centers[-i] = nxt
        cur = nxt
    return centers


def _expand_selection_axis(
    context,
    long_axis: Vector,
    cross_axis: Vector,
    flip: bool,
    axis_label: str,
) -> tuple[bool, str | None]:
    """Expandir seleccion a 20 verts usando ejes long/cross."""
    obj = context.edit_object
    if obj is None or obj.type != "MESH" or context.mode != "EDIT_MESH":
        return False, "Info: activa un MESH en Edit Mode."

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    active = bm.select_history.active
    if not isinstance(active, bmesh.types.BMVert) or not active.is_valid:
        active = None
        for elem in reversed(list(bm.select_history)):
            if isinstance(elem, bmesh.types.BMVert) and elem.is_valid:
                active = elem
                break
        if active is None:
            active = next((v for v in bm.verts if v.select and v.is_valid), None)
    if active is None:
        return False, "Aviso: selecciona un vertice activo."

    if flip:
        offsets = [-3, -2, -1, 0]
    else:
        offsets = [0, 1, 2, 3]

    centers = _build_centers(active, cross_axis, offsets)
    if centers is None:
        return False, f"Aviso: no se pudo expandir en el eje {axis_label}."

    rails: list[list[bmesh.types.BMVert]] = []
    for off in offsets:
        rail = _build_centered_rail(centers[off], long_axis, AUTO_SELECT_LENGTH)
        if rail is None:
            return False, f"Aviso: rail demasiado corto en eje {axis_label}."
        rails.append(rail)

    selected = {v for rail in rails for v in rail}
    if len(selected) != AUTO_SELECT_RAILS * AUTO_SELECT_LENGTH:
        return False, "Aviso: la seleccion no llego a 20 vertices."

    for v in bm.verts:
        v.select = False
    for v in selected:
        v.select = True
    bm.select_history.clear()
    bm.select_history.add(active)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return True, None


def expand_selection_mouse(context, event) -> tuple[bool, str | None]:
    """Auto-seleccionar 20 verts usando direccion del mouse."""
    obj = context.edit_object
    if obj is None or obj.type != "MESH" or context.mode != "EDIT_MESH":
        return False, "Info: activa un MESH en Edit Mode."

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    active = bm.select_history.active
    if not isinstance(active, bmesh.types.BMVert) or not active.is_valid:
        active = None
        for elem in reversed(list(bm.select_history)):
            if isinstance(elem, bmesh.types.BMVert) and elem.is_valid:
                active = elem
                break
        if active is None:
            active = next((v for v in bm.verts if v.select and v.is_valid), None)
    if active is None:
        return False, "Aviso: selecciona un vertice activo."

    # Limpiar seleccion a solo el activo.
    for v in bm.verts:
        v.select = (v == active)
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    bm.select_history.clear()
    bm.select_history.add(active)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    axes = _pick_axes(active)
    if len(axes) < 2:
        return False, "Aviso: no se detectaron dos ejes en la seleccion."

    mouse_x = None
    mouse_y = None
    if event is not None:
        mouse_x, mouse_y = event.mouse_x, event.mouse_y
        if context.window_manager is not None:
            context.window_manager[MOUSE_CACHE_KEY] = (mouse_x, mouse_y)
    elif context.window_manager is not None and MOUSE_CACHE_KEY in context.window_manager:
        mouse_x, mouse_y = context.window_manager[MOUSE_CACHE_KEY]
    else:
        win = context.window
        if win is not None:
            try:
                mouse_x = win.event_mouse_x
                mouse_y = win.event_mouse_y
            except Exception:
                mouse_x = None
                mouse_y = None
        if (mouse_x is None or mouse_y is None) and context.window_manager is not None:
            for win in context.window_manager.windows:
                try:
                    mouse_x = win.event_mouse_x
                    mouse_y = win.event_mouse_y
                    break
                except Exception:
                    continue

    region, rv3d = _find_view3d_region(context, mouse_x, mouse_y)
    if region is None or rv3d is None:
        return False, "Aviso: no se pudo acceder al Viewport 3D."

    if mouse_x is None or mouse_y is None:
        coord = (region.width * 0.5, region.height * 0.5)
    else:
        coord = (mouse_x - region.x, mouse_y - region.y)
        if (
            coord[0] < 0
            or coord[1] < 0
            or coord[0] > region.width
            or coord[1] > region.height
        ):
            coord = (region.width * 0.5, region.height * 0.5)

    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

    view_normal = rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
    active_world = obj.matrix_world @ active.co
    hit = geometry.intersect_line_plane(
        origin, origin + direction * 1000.0, active_world, view_normal, False
    )
    if hit is None:
        hit = origin + direction
    to_cursor_world = hit - active_world
    if to_cursor_world.length < 1e-6:
        return False, "Aviso: el mouse esta demasiado cerca del vertice activo."

    to_cursor = obj.matrix_world.inverted().to_3x3() @ to_cursor_world
    if to_cursor.length < 1e-6:
        return False, "Aviso: direccion de mouse invalida."
    to_cursor.normalize()

    axis_a, axis_b = axes[0], axes[1]
    if abs(axis_b.dot(to_cursor)) > abs(axis_a.dot(to_cursor)):
        cross_axis = axis_b
        long_axis = axis_a
    else:
        cross_axis = axis_a
        long_axis = axis_b

    flip = to_cursor.dot(cross_axis) < 0.0
    return _expand_selection_axis(context, long_axis, cross_axis, flip, "Mouse")


def detect_active_vert(context):
    """Detectar vert activo y guardar info en custom props del objeto."""
    obj = context.edit_object
    if obj is None or obj.type != "MESH" or context.mode != "EDIT_MESH":
        return None, "Info: activa un MESH en Edit Mode."

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    active = bm.select_history.active
    if not isinstance(active, bmesh.types.BMVert) or not active.is_valid:
        return None, "Aviso: no hay vertice activo."

    info = {
        "index": int(active.index),
        "co": (float(active.co.x), float(active.co.y), float(active.co.z)),
    }
    obj["sdm_active_vert_index"] = info["index"]
    obj["sdm_active_vert_co"] = info["co"]
    return info, None
