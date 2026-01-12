"""Operadores del addon Simple Diamond."""

import bpy
from bpy.props import BoolProperty, FloatProperty

from . import core, pre_step


class SDM_OT_auto_select_cursor(bpy.types.Operator):
    """Auto-seleccion por mouse para 20 verts."""

    bl_idname = "mesh.sdm_auto_select_cursor"
    bl_label = "Auto Select Mouse_SimpleDiamond"
    bl_description = "Expande la seleccion de 20 verts hacia el mouse"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode == "EDIT_MESH"

    def execute(self, context):
        ok, msg = pre_step.expand_selection_mouse(context, None)
        if msg:
            self.report({"INFO"}, msg)
        if not ok:
            return {"CANCELLED"}
        return {"FINISHED"}

    def invoke(self, context, event):
        ok, msg = pre_step.expand_selection_mouse(context, event)
        if msg:
            self.report({"INFO"}, msg)
        if not ok:
            return {"CANCELLED"}
        return {"FINISHED"}


class SDM_OT_run(bpy.types.Operator):
    """Operador principal de Simple Diamond."""

    bl_idname = "mesh.sdm_simple_diamond"
    bl_label = "Simple Diamond_SimpleDiamond"
    bl_description = "Run Simple Diamond on the current selection"
    bl_options = {"REGISTER", "UNDO"}

    invert_connection: BoolProperty(
        name="Invert",
        description="Usa la logica invertida",
        default=core.INVERT_CONNECTION,
    )
    alignment_threshold: FloatProperty(
        name="Rail Threshold",
        description="Umbral de alineacion para detectar rails",
        default=core.ALIGNMENT_THRESHOLD,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    slide_factor: FloatProperty(
        name="Slide Factor",
        description="Factor de desplazamiento del vertex slide",
        default=core.SLIDE_FACTOR,
        min=0.0,
    )
    slide_clamp: BoolProperty(
        name="Clamp Slide",
        description="Limita el slide a la longitud del edge",
        default=core.SLIDE_CLAMP,
    )
    verify_connections: BoolProperty(
        name="Verify Connections",
        description="Verifica las conexiones esperadas",
        default=core.VERIFY_CONNECTIONS,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode == "EDIT_MESH"

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "invert_connection")
        layout.prop(self, "alignment_threshold")
        layout.prop(self, "slide_factor")
        layout.prop(self, "slide_clamp")
        layout.prop(self, "verify_connections")

    def execute(self, context):
        try:
            info, msg = pre_step.detect_active_vert(context)
            if msg:
                self.report({"WARNING"}, msg)
            elif info is not None:
                self.report({"INFO"}, f"Active vert: {info['index']}")
            core.INVERT_CONNECTION = self.invert_connection
            core.ALIGNMENT_THRESHOLD = self.alignment_threshold
            core.SLIDE_FACTOR = self.slide_factor
            core.SLIDE_CLAMP = self.slide_clamp
            core.VERIFY_CONNECTIONS = self.verify_connections
            core.run()
        except Exception as exc:
            self.report({"ERROR"}, f"Simple Diamond failed: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


CLASSES = (SDM_OT_auto_select_cursor, SDM_OT_run)


def register():
    """Registrar clases del modulo ops."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    """Desregistrar clases del modulo ops."""
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
