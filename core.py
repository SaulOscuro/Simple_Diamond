"""Logica central del addon Simple Diamond (20 verts, 4 rails x 5).

Requisitos:
- Objeto activo MESH en Edit Mode.
- Exactamente 20 verts seleccionados con edges entre ellos.

Efectos:
- Crea edges entre rails, aplica slide y hace loop dissolve.
- Usa bpy.ops con seleccion temporal; restaura la seleccion original.

Flags:
- INVERT_CONNECTION alterna el modo normal/invert.
- ALIGNMENT_THRESHOLD ajusta deteccion de rails.
- SLIDE_FACTOR y SLIDE_CLAMP controlan el slide.
- VERIFY_CONNECTIONS valida edges esperados.
"""

from __future__ import annotations

import bpy
import bmesh
from mathutils import Vector

ALIGNMENT_THRESHOLD = 0.55
INVERT_CONNECTION = False
SELECT_CREATED_EDGES = True
DEBUG_PRINT = False
VERIFY_CONNECTIONS = True
SLIDE_FACTOR = 0.5
SLIDE_CLAMP = True

# Indices 1-based para conexiones y slide.
CONNECT_B_TO_C = (2, 4)
CONNECT_C_TARGET = 3
CONNECT_C_TO_B = (2, 4)
CONNECT_B_TARGET = 3
SLIDE_FROM_B = 3
SLIDE_TO_A = 3
SLIDE_FROM_C = 3
SLIDE_TO_D = 3

# Pares para loop dissolve (1-based).
LOOP_B2 = 2
LOOP_C2 = 2
LOOP_B4 = 4
LOOP_C4 = 4
DISSOLVE_B3 = 3
DISSOLVE_C3 = 3


def farthest_pair(points: list[Vector]) -> tuple[int, int]:
    """Buscar el par mas lejano (O(n^2))."""
    best_i, best_j = 0, 0
    best_d2 = -1.0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (points[j] - points[i]).length_squared
            if d2 > best_d2:
                best_d2 = d2
                best_i, best_j = i, j
    return best_i, best_j


def choose_axis_from_verts(verts: list[bmesh.types.BMVert]) -> Vector:
    """Elegir eje primario desde el par mas lejano."""
    pts = [v.co.copy() for v in verts]
    i, j = farthest_pair(pts)
    axis = pts[j] - pts[i]
    if axis.length < 1e-12:
        axis = Vector((1.0, 0.0, 0.0))
    else:
        axis.normalize()
    return axis


def aligned_edges(
    edges: list[bmesh.types.BMEdge],
    axis: Vector,
    threshold: float,
) -> list[bmesh.types.BMEdge]:
    """Filtrar edges alineados al eje principal."""
    out: list[bmesh.types.BMEdge] = []
    for e in edges:
        d = e.verts[1].co - e.verts[0].co
        if d.length < 1e-12:
            continue
        d.normalize()
        if abs(d.dot(axis)) >= threshold:
            out.append(e)
    return out


def build_edge_components(edges: list[bmesh.types.BMEdge]) -> list[set]:
    """Construir componentes conectados de edges candidatos a rail."""
    v_to_edges: dict[bmesh.types.BMVert, list[bmesh.types.BMEdge]] = {}
    for e in edges:
        for v in e.verts:
            v_to_edges.setdefault(v, []).append(e)

    seen: set[bmesh.types.BMEdge] = set()
    comps = []
    for e0 in edges:
        if e0 in seen:
            continue
        stack = [e0]
        comp: set[bmesh.types.BMEdge] = set()
        seen.add(e0)

        while stack:
            e = stack.pop()
            comp.add(e)
            for v in e.verts:
                for ne in v_to_edges.get(v, []):
                    if ne not in seen:
                        seen.add(ne)
                        stack.append(ne)

        comps.append(comp)
    return comps


def order_chain_vertices(
    edge_set: set[bmesh.types.BMEdge],
    axis: Vector,
) -> list[bmesh.types.BMVert]:
    """Ordenar verts de un rail por proyeccion en el eje."""
    nbrs: dict[bmesh.types.BMVert, list[bmesh.types.BMVert]] = {}
    for e in edge_set:
        a, b = e.verts
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)

    verts = list(nbrs.keys())
    if not verts:
        return []

    endpoints = [v for v in verts if len(nbrs[v]) == 1]
    if endpoints:
        start = min(endpoints, key=lambda v: v.co.dot(axis))
        is_loop = False
    else:
        start = min(verts, key=lambda v: v.co.dot(axis))
        is_loop = True

    ordered: list[bmesh.types.BMVert] = []
    prev = None
    cur = start
    max_steps = len(verts) + 10

    for _ in range(max_steps):
        if not cur.is_valid:
            break
        ordered.append(cur)
        candidates = [n for n in nbrs[cur] if n != prev and n.is_valid]
        if not candidates:
            break
        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            cur_proj = cur.co.dot(axis)
            nxt = max(candidates, key=lambda n: n.co.dot(axis) - cur_proj)
        prev, cur = cur, nxt
        if is_loop and cur == start:
            break

    seen: set[bmesh.types.BMVert] = set()
    out: list[bmesh.types.BMVert] = []
    for v in ordered:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _align_rail_to_reference(
    reference: list[bmesh.types.BMVert],
    rail: list[bmesh.types.BMVert],
) -> list[bmesh.types.BMVert]:
    """Invertir rail si queda en direccion opuesta al rail referencia."""
    if not reference or not rail:
        return rail
    ref_start = reference[0].co
    ref_end = reference[-1].co
    r0 = rail[0].co
    r1 = rail[-1].co
    dist_same = (ref_start - r0).length + (ref_end - r1).length
    dist_flip = (ref_start - r1).length + (ref_end - r0).length
    if dist_flip < dist_same:
        return list(reversed(rail))
    return rail


def _secondary_axis_index(axis: Vector, verts: list[bmesh.types.BMVert]) -> int:
    """Elegir eje secundario por varianza (excluye eje primario)."""
    primary = max(range(3), key=lambda i: abs(axis[i]))
    coords = [(v.co.x, v.co.y, v.co.z) for v in verts]
    variances = []
    for i in range(3):
        vals = [c[i] for c in coords]
        mean = sum(vals) / len(vals)
        variances.append(sum((v - mean) ** 2 for v in vals))
    variances[primary] = -1.0
    return max(range(3), key=lambda i: variances[i])


def _order_verts_by_axis(
    verts: list[bmesh.types.BMVert],
    axis: Vector,
) -> list[bmesh.types.BMVert]:
    """Ordenar verts por proyeccion del eje."""
    return sorted(verts, key=lambda v: v.co.dot(axis))


def _group_rails_by_secondary(
    verts: list[bmesh.types.BMVert],
    axis: Vector,
    secondary_idx: int,
    expected_groups: int,
    expected_size: int,
) -> list[list[bmesh.types.BMVert]] | None:
    """Agrupar verts en rails usando eje secundario."""
    if not verts:
        return None

    items = sorted(verts, key=lambda v: v.co[secondary_idx])

    rounded: dict[float, list[bmesh.types.BMVert]] = {}
    for v in items:
        key = round(v.co[secondary_idx], 6)
        rounded.setdefault(key, []).append(v)
    if len(rounded) == expected_groups:
        groups = list(rounded.values())
        groups.sort(key=lambda g: sum(v.co[secondary_idx] for v in g) / len(g))
        return [_order_verts_by_axis(g, axis) for g in groups]

    coords = [v.co[secondary_idx] for v in items]
    gaps = [(coords[i + 1] - coords[i], i) for i in range(len(coords) - 1)]
    gaps.sort(key=lambda g: g[0], reverse=True)
    if len(gaps) >= expected_groups - 1:
        split_idxs = sorted([idx + 1 for _, idx in gaps[: expected_groups - 1]])
        groups = []
        start = 0
        for idx in split_idxs:
            groups.append(items[start:idx])
            start = idx
        groups.append(items[start:])
        if len(groups) == expected_groups:
            groups.sort(key=lambda g: sum(v.co[secondary_idx] for v in g) / len(g))
            return [_order_verts_by_axis(g, axis) for g in groups]

    if len(items) == expected_groups * expected_size:
        groups = [
            items[i * expected_size : (i + 1) * expected_size]
            for i in range(expected_groups)
        ]
        groups.sort(key=lambda g: sum(v.co[secondary_idx] for v in g) / len(g))
        return [_order_verts_by_axis(g, axis) for g in groups]

    return None


def _index_to_zero(idx: int) -> int:
    """Convertir indice 1-based a 0-based."""
    return idx - 1


def _as_edge(item: object) -> bmesh.types.BMEdge | None:
    """Normalizar retorno a BMEdge cuando un op devuelve wrappers."""
    if isinstance(item, bmesh.types.BMEdge):
        return item
    if isinstance(item, bmesh.types.BMLoop):
        return item.edge
    edge = getattr(item, "edge", None)
    if isinstance(edge, bmesh.types.BMEdge):
        return edge
    return None


def _find_edge(
    v1: bmesh.types.BMVert,
    v2: bmesh.types.BMVert,
) -> bmesh.types.BMEdge | None:
    """Buscar edge directo entre dos verts."""
    for e in v1.link_edges:
        if (e.verts[0] is v1 and e.verts[1] is v2) or (e.verts[0] is v2 and e.verts[1] is v1):
            return e
    return None


def _dissolve_edge_single(
    bm: bmesh.types.BMesh,
    v1: bmesh.types.BMVert,
    v2: bmesh.types.BMVert,
) -> bool:
    """Disolver un edge directo entre dos verts."""
    edge = _find_edge(v1, v2)
    if edge is None or not edge.is_valid:
        return False
    try:
        bmesh.ops.dissolve_edges(bm, edges=[edge])
    except Exception:
        return False
    return True


def slide_vert_toward(
    vert: bmesh.types.BMVert,
    target: bmesh.types.BMVert,
    clamp: bool,
    factor: float,
) -> bool:
    """Deslizar un vert a lo largo de su edge mas alineado al target."""
    if not vert.is_valid or not target.is_valid:
        return False
    to_target = target.co - vert.co
    if to_target.length < 1e-8:
        return False
    to_target_dir = to_target.normalized()

    best_edge = None
    best_dot = -1.0
    best_len = 0.0
    best_dir = None
    for e in vert.link_edges:
        other = e.other_vert(vert)
        if not other.is_valid:
            continue
        edge_vec = other.co - vert.co
        if edge_vec.length < 1e-8:
            continue
        edge_dir = edge_vec.normalized()
        dot = edge_dir.dot(to_target_dir)
        if dot > best_dot:
            best_dot = dot
            best_edge = e
            best_len = edge_vec.length
            best_dir = edge_dir

    if best_edge is None or best_dir is None or best_dot <= 0.0:
        return False

    proj = best_dir.dot(to_target)
    if proj <= 0.0:
        return False
    dist = proj * factor
    if clamp:
        dist = min(dist, best_len)

    vert.co = vert.co + best_dir * dist
    return True


def _op_connect_path(
    bm: bmesh.types.BMesh,
    obj: bpy.types.Object,
    v1: bmesh.types.BMVert,
    v2: bmesh.types.BMVert,
) -> list[bmesh.types.BMEdge]:
    """Conectar verts via bpy.ops.mesh.vert_connect_path.

    Guarda y restaura la seleccion y el mesh_select_mode.
    """
    # Guardar seleccion actual para restaurarla despues.
    sel_verts = [v for v in bm.verts if v.select]
    sel_edges = [e for e in bm.edges if e.select]
    sel_faces = [f for f in bm.faces if f.select]

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    before_pairs = {
        tuple(sorted((e.verts[0].index, e.verts[1].index)))
        for e in bm.edges
        if e.is_valid
    }

    # Seleccionar solo los verts objetivo.
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    v1.select = True
    v2.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    view_layer = bpy.context.view_layer
    if view_layer is not None:
        view_layer.objects.active = obj

    tool_settings = bpy.context.tool_settings
    prev_select_mode = tuple(tool_settings.mesh_select_mode)
    tool_settings.mesh_select_mode = (True, False, False)

    # Usar override de VIEW_3D cuando sea posible.
    override = None
    screen = bpy.context.window.screen if bpy.context.window else None
    if screen is not None:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is not None:
                override = {"area": area, "region": region, "active_object": obj, "object": obj}
                break

    try:
        if override:
            with bpy.context.temp_override(**override):
                result = bpy.ops.mesh.vert_connect_path()
        else:
            result = bpy.ops.mesh.vert_connect_path()
        if DEBUG_PRINT and result != {"FINISHED"}:
            print(f"Connect path result: {result}")
    except Exception as exc:
        if DEBUG_PRINT:
            print(f"Connect path failed: {exc}")
    finally:
        tool_settings.mesh_select_mode = prev_select_mode

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    after_pairs = {
        tuple(sorted((e.verts[0].index, e.verts[1].index)))
        for e in bm.edges
        if e.is_valid
    }
    new_pairs = after_pairs - before_pairs
    new_edges: list[bmesh.types.BMEdge] = []
    if new_pairs:
        for e in bm.edges:
            if not e.is_valid:
                continue
            pair = tuple(sorted((e.verts[0].index, e.verts[1].index)))
            if pair in new_pairs:
                new_edges.append(e)

    # Restaurar seleccion original.
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    for v in sel_verts:
        if v.is_valid:
            v.select = True
    for e in sel_edges:
        if e.is_valid:
            e.select = True
    for f in sel_faces:
        if f.is_valid:
            f.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    return new_edges


def _op_dissolve_edge_loop(
    bm: bmesh.types.BMesh,
    obj: bpy.types.Object,
    v1: bmesh.types.BMVert,
    v2: bmesh.types.BMVert,
) -> bool:
    """Disolver un loop de edges a partir de un edge seleccionado."""
    # Guardar seleccion actual para restaurarla despues.
    sel_verts = [v for v in bm.verts if v.select]
    sel_edges = [e for e in bm.edges if e.select]
    sel_faces = [f for f in bm.faces if f.select]

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    edge = _find_edge(v1, v2)
    if edge is None:
        return False

    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    edge.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    view_layer = bpy.context.view_layer
    if view_layer is not None:
        view_layer.objects.active = obj

    tool_settings = bpy.context.tool_settings
    prev_select_mode = tuple(tool_settings.mesh_select_mode)
    tool_settings.mesh_select_mode = (False, True, False)

    # Usar override de VIEW_3D cuando sea posible.
    override = None
    screen = bpy.context.window.screen if bpy.context.window else None
    if screen is not None:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is not None:
                override = {"area": area, "region": region, "active_object": obj, "object": obj}
                break

    ok = True
    try:
        if override:
            with bpy.context.temp_override(**override):
                result = bpy.ops.mesh.loop_multi_select(ring=False)
                if result != {"FINISHED"}:
                    ok = False
                result = bpy.ops.mesh.dissolve_edges()
                if result != {"FINISHED"}:
                    ok = False
        else:
            result = bpy.ops.mesh.loop_multi_select(ring=False)
            if result != {"FINISHED"}:
                ok = False
            result = bpy.ops.mesh.dissolve_edges()
            if result != {"FINISHED"}:
                ok = False
    except Exception as exc:
        if DEBUG_PRINT:
            print(f"Loop dissolve failed: {exc}")
        ok = False
    finally:
        tool_settings.mesh_select_mode = prev_select_mode

    # Restaurar seleccion original.
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    for v in sel_verts:
        if v.is_valid:
            v.select = True
    for e in sel_edges:
        if e.is_valid:
            e.select = True
    for f in sel_faces:
        if f.is_valid:
            f.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    return ok


def connect_pair(
    bm: bmesh.types.BMesh,
    obj: bpy.types.Object,
    v1: bmesh.types.BMVert,
    v2: bmesh.types.BMVert,
) -> tuple[str, list[bmesh.types.BMEdge]]:
    """Conectar dos verts con un edge directo si no existe."""
    existing = _find_edge(v1, v2)
    if existing is not None:
        return "existing", [existing]
    new_edges = _op_connect_path(bm, obj, v1, v2)
    if new_edges:
        return "created", new_edges
    return "failed", []


def run() -> None:
    """Ejecutar la logica Simple Diamond sobre la seleccion actual."""
    # --- Paso 1: contexto y validacion ---
    obj = bpy.context.edit_object
    if obj is None or obj.type != "MESH" or bpy.context.mode != "EDIT_MESH":
        print("Info: activa un objeto MESH en Edit Mode.")
        return

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    selected_verts = [v for v in bm.verts if v.select]
    if len(selected_verts) != 20:
        print("Aviso: deben ser 20 vertices seleccionados.")
        return

    candidate_edges = [e for e in bm.edges if e.verts[0].select and e.verts[1].select]
    if not candidate_edges:
        print("Aviso: no hay edges entre los vertices seleccionados.")
        return

    # --- Paso 2: deteccion de rails ---
    axis = choose_axis_from_verts(selected_verts)
    secondary_idx = _secondary_axis_index(axis, selected_verts)
    rail_edges = aligned_edges(candidate_edges, axis, ALIGNMENT_THRESHOLD)
    if not rail_edges:
        print("Aviso: no se detectaron edges alineados a rails.")
        return

    comps = build_edge_components(rail_edges)
    if len(comps) >= 4:
        comps.sort(key=lambda s: len(s), reverse=True)
        rails = [order_chain_vertices(c, axis) for c in comps[:4]]
    else:
        rails = _group_rails_by_secondary(
            selected_verts,
            axis,
            secondary_idx,
            expected_groups=4,
            expected_size=5,
        )
        if rails is None:
            print("Aviso: no se detectaron cuatro rails.")
            return

    if any(len(r) < 5 for r in rails):
        print("Aviso: rails demasiado cortos para 5 vertices.")
        return

    # --- Paso 3: orden y alineacion ---
    def _rail_mean(items: list[bmesh.types.BMVert]) -> float:
        return sum(v.co[secondary_idx] for v in items) / len(items)

    active = bm.select_history.active
    if isinstance(active, bmesh.types.BMVert) and active in selected_verts:
        active_rail = None
        for r in rails:
            if active in r:
                active_rail = r
                break
        if active_rail is not None:
            remaining = [r for r in rails if r is not active_rail]
            a_mean = _rail_mean(active_rail)
            remaining.sort(key=lambda r: _rail_mean(r) - a_mean)
            A = active_rail
            B, C, D = remaining[0], remaining[1], remaining[2]
        else:
            rails.sort(key=_rail_mean)
            A, B, C, D = rails[0], rails[1], rails[2], rails[3]
    else:
        rails.sort(key=_rail_mean)
        A, B, C, D = rails[0], rails[1], rails[2], rails[3]

    B = _align_rail_to_reference(A, B)
    C = _align_rail_to_reference(A, C)
    D = _align_rail_to_reference(A, D)

    # --- Paso 4: conexiones ---
    b_indices = [_index_to_zero(i) for i in CONNECT_B_TO_C]
    c_target = _index_to_zero(CONNECT_C_TARGET)
    c_indices = [_index_to_zero(i) for i in CONNECT_C_TO_B]
    b_target = _index_to_zero(CONNECT_B_TARGET)

    if INVERT_CONNECTION:
        if max(c_indices) >= len(C) or b_target >= len(B):
            print("Aviso: indices fuera de rango para C/B.")
            return
    else:
        if max(b_indices) >= len(B) or c_target >= len(C):
            print("Aviso: indices fuera de rango para B/C.")
            return

    connection_pairs: list[tuple[str, bmesh.types.BMVert, bmesh.types.BMVert]] = []
    if INVERT_CONNECTION:
        for ci in c_indices:
            v1 = C[ci]
            v2 = B[b_target]
            if not v1.is_valid or not v2.is_valid:
                continue
            connection_pairs.append((f"C{ci + 1} -> B{b_target + 1}", v1, v2))
    else:
        for bi in b_indices:
            v1 = B[bi]
            v2 = C[c_target]
            if not v1.is_valid or not v2.is_valid:
                continue
            connection_pairs.append((f"B{bi + 1} -> C{c_target + 1}", v1, v2))

    made = 0
    existing = 0
    failed = 0
    created_edges: list[bmesh.types.BMEdge] = []
    connection_results: list[
        tuple[str, bmesh.types.BMVert, bmesh.types.BMVert, str, list[bmesh.types.BMEdge]]
    ] = []
    for label, v1, v2 in connection_pairs:
        status, edges = connect_pair(bm, obj, v1, v2)
        if status == "created":
            made += 1
        elif status == "existing":
            existing += 1
        else:
            failed += 1
        created_edges.extend(edges)
        connection_results.append((label, v1, v2, status, edges))

    # --- Paso 5: disolver edge B3/C3 ---
    b3_idx = _index_to_zero(DISSOLVE_B3)
    c3_idx = _index_to_zero(DISSOLVE_C3)
    if b3_idx < len(B) and c3_idx < len(C):
        if not _dissolve_edge_single(bm, B[b3_idx], C[c3_idx]):
            print("Aviso: no se pudo disolver el edge B3/C3.")
    else:
        print("Aviso: indices fuera de rango para disolver B3/C3.")

    # --- Paso 6: slide ---
    if INVERT_CONNECTION:
        slide_from_idx = _index_to_zero(SLIDE_FROM_C)
        slide_to_idx = _index_to_zero(SLIDE_TO_D)
        if slide_from_idx >= len(C) or slide_to_idx >= len(D):
            print("Aviso: indices fuera de rango para slide C/D.")
        else:
            slide_from = C[slide_from_idx]
            slide_to = D[slide_to_idx]
            if not slide_vert_toward(slide_from, slide_to, SLIDE_CLAMP, SLIDE_FACTOR):
                print("Aviso: slide fallido (C3->D3).")
    else:
        slide_from_idx = _index_to_zero(SLIDE_FROM_B)
        slide_to_idx = _index_to_zero(SLIDE_TO_A)
        if slide_from_idx >= len(B) or slide_to_idx >= len(A):
            print("Aviso: indices fuera de rango para slide B/A.")
        else:
            slide_from = B[slide_from_idx]
            slide_to = A[slide_to_idx]
            if not slide_vert_toward(slide_from, slide_to, SLIDE_CLAMP, SLIDE_FACTOR):
                print("Aviso: slide fallido (B3->A3).")

    # --- Paso 7: loop dissolve ---
    def _dissolve_step(label: str, v1: bmesh.types.BMVert, v2: bmesh.types.BMVert) -> None:
        if not v1.is_valid or not v2.is_valid:
            print(f"Loop dissolve SKIP: {label} (missing vertex)")
            return
        ok = _op_dissolve_edge_loop(bm, obj, v1, v2)
        if ok:
            print(f"Loop dissolve OK: {label}")
        else:
            print(f"Loop dissolve FAIL: {label}")

    b2_idx = _index_to_zero(LOOP_B2)
    c2_idx = _index_to_zero(LOOP_C2)
    b4_idx = _index_to_zero(LOOP_B4)
    c4_idx = _index_to_zero(LOOP_C4)
    if b2_idx < len(B) and c2_idx < len(C):
        label = "B2/C2" if not INVERT_CONNECTION else "C2/B2"
        _dissolve_step(label, B[b2_idx], C[c2_idx])
    if b4_idx < len(B) and c4_idx < len(C):
        label = "B4/C4" if not INVERT_CONNECTION else "C4/B4"
        _dissolve_step(label, B[b4_idx], C[c4_idx])

    # --- Paso 8: verificacion ---
    if VERIFY_CONNECTIONS:
        missing = 0
        for label, v1, v2, status, _edges in connection_results:
            if status in {"created", "existing"}:
                continue
            if not v1.is_valid or not v2.is_valid:
                print(f"Missing edge (invalid verts): {label}")
                missing += 1
                continue
            if _find_edge(v1, v2) is None:
                print(f"Missing edge: {label}")
                missing += 1
        if missing == 0:
            print("Verify: all expected edges present.")

    # --- Paso 9: seleccion final y update ---
    if SELECT_CREATED_EDGES:
        for item in created_edges:
            edge = _as_edge(item)
            if edge is not None and edge.is_valid:
                edge.select = True

    bm.normal_update()
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
    mode_label = "invert" if INVERT_CONNECTION else "normal"
    print(
        f"OK: connected {made} edges, existing {existing}, failed {failed} ({mode_label})."
    )


if __name__ == "__main__":
    run()
