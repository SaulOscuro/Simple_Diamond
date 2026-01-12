# Simple Diamond

**Simple Diamond** es un complemento (addon) para Blender 4.x que automatiza la creación de estructuras de tipo "diamond loop" a partir de una selección de 20 vértices en modo edición. Forma parte del proyecto Loop_Reduction.

## Descripción
El algoritmo conecta los cuatro rails de cinco vértices que forman un ciclo, aplica un slide de vértices y realiza un loop dissolve para reducir el bucle sin perder la topología. Todo se ejecuta mediante operadores de Blender y la selección original se restaura tras completarse.

Entre las características destacan:
- Selección automática por ratón de los 20 vértices iniciales.
- Ajuste de la tolerancia de alineación para detectar los rails.
- Posibilidad de invertir el sentido de las conexiones.
- Verificación de conexiones esperadas y opción para hacer clamp del slide.
- Actualizaciones graduadas en las versiones 0.2 y 0.3 para mejorar el flujo de trabajo.

## Instalación
1. Descarga este repositorio o clónalo con `git clone`.
2. En Blender ve a *Edit > Preferences > Add-ons > Install* y selecciona el archivo ZIP del addon o la carpeta `simple_diamond_addon`.
3. Activa el addon y aparecerán dos operadores en el menú de mallas (*Mesh > Simple Diamond*).

## Uso
1. Selecciona un objeto de tipo MESH y entra en modo edición.
2. Selecciona 20 vértices que formen un patrón de diamante o usa el operador "Auto Select" para que el complemento amplíe la selección alrededor del cursor.
3. Ejecuta el operador *Simple Diamond* desde el menú. Puedes ajustar las propiedades antes de confirmar (invertir conexiones, umbral de alineación, factor de slide, clamp, verificación de conexiones).
4. El addon conectará los rails, aplicará el desplazamiento de vértices y realizará el loop dissolve; la selección original se restablecerá al final.

## Historial de versiones
- **0.1**: Primer lanzamiento. Incluye lógica básica de conexiones, slide y loop dissolve.
- **0.2**: Añade disolución específica del borde B3/C3 y reordena los pasos para mayor estabilidad.
- **0.3**: Mejora el ordenamiento de rails calculando centros y una dirección global para ordenar los vértices.

## Licencia
Indica la licencia aplicable al addon (por ejemplo MIT o GPL). Asegúrate de completarla si el proyecto está destinado a distribuirse públicamente.
