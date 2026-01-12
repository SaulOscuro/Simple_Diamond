"""UI del addon Simple Diamond."""

import bpy

from . import ops


def menu_func(self, context):
    """Agregar operadores al menu de mallas."""
    layout = self.layout
    layout.operator_context = "INVOKE_DEFAULT"
    layout.operator(ops.SDM_OT_auto_select_cursor.bl_idname, icon="VERTEXSEL")
    layout.operator_context = "EXEC_DEFAULT"
    layout.operator(ops.SDM_OT_run.bl_idname, icon="MESH_GRID")


def register():
    """Registrar menu."""
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    """Desregistrar menu."""
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
