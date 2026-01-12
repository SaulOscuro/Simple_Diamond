"""Addon Simple Diamond.

Requiere Blender 4.x. Expone operadores para auto-seleccion (20 verts)
y para ejecutar la logica de conexiones/slide.
"""

bl_info = {
    "name": "Simple Diamond",
    "author": "User",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Mesh",
    "description": "Simple diamond tool (normal/invert) for 20 selected verts",
    "category": "Mesh",
}

from . import ops
from . import ui


def register():
    """Registrar clases del addon."""
    ops.register()
    ui.register()


def unregister():
    """Desregistrar clases del addon."""
    ui.unregister()
    ops.unregister()
