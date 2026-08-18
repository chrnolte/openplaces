"""
QGIS processing algorithm (plugin) to load an `openplaces` parquet
delivery and join its parts into one layer.

A delivery is a set of files sharing one stem and one index:

    {base}.parquet            canonical attributes (no geometry)
    {base}_geo.parquet        boundary polygons
    {base}_geo_simplified.parquet   coarser polygons, for fast rendering
    {base}_point.parquet      canonical attributes on centroid points
    {base}_evidence.parquet   every remaining column

Pick any one of them as the input; the rest are resolved by name.
The parts are joined on '_join_id', 'geo_id', or whichever entity id
column the files share (e.g. 'footprint_id').

The older two-file layout (attributes + '_geo' sidecar, no '_point' or
'_evidence') loads unchanged.
"""

from pathlib import Path

from qgis.core import (
    Qgis,
    QgsProcessingAlgorithm,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProject,
    QgsVectorLayer,
    QgsVectorLayerJoinInfo,
)

# Suffixes marking a member of a delivery rather than part of its
# name. Longest first, so '_geo_simplified' is not read as '_geo'.
PART_SUFFIXES = ('_geo_simplified', '_geo', '_point', '_evidence')

# Geometry choices, in the order shown in the dialog.
GEOMETRY_POLYGONS = 0
GEOMETRY_POINTS = 1

# Never a join key: written by GeoParquet's covering-bbox option and by
# the geometry column itself.
NON_KEY_FIELDS = ('bbox', 'geometry')


class LoadJoinedParquetAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    GEOMETRY = 'GEOMETRY'
    USE_SIMPLIFIED = 'USE_SIMPLIFIED'
    JOIN_EVIDENCE = 'JOIN_EVIDENCE'

    def createInstance(self):
        return LoadJoinedParquetAlgorithm()

    def name(self):
        return 'loadjoinedparquet'

    def displayName(self):
        return 'Load joined `openplaces` parquet files'

    def group(self):
        return 'openplaces'

    def groupId(self):
        return 'openplaces'

    def shortHelpString(self):
        return (
            'Load an openplaces parquet delivery and join its parts into one '
            'layer. Pick any file of the set; the others are found by name.\n\n'
            'Geometry: "Boundary polygons" renders the *_geo.parquet outlines '
            'and joins the canonical attributes onto them. "Centroid points" '
            'renders *_point.parquet instead, which already carries those '
            'attributes: much faster, and enough for symbolizing values.\n\n'
            '"Use simplified geometries" prefers a *_geo_simplified.parquet '
            'file for faster polygon rendering when one exists.\n\n'
            '"Join evidence columns" also joins *_evidence.parquet, the '
            'supplement holding every column the canonical file leaves out.'
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT, 'Any file of the parquet delivery', extension='parquet'
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.GEOMETRY,
                'Geometry',
                options=['Boundary polygons', 'Centroid points'],
                defaultValue=GEOMETRY_POLYGONS,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_SIMPLIFIED,
                'Use simplified geometries (*_geo_simplified.parquet) if available',
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.JOIN_EVIDENCE,
                'Join evidence columns (*_evidence.parquet) if available',
                defaultValue=False,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        input_file = Path(self.parameterAsFile(parameters, self.INPUT, context))
        geometry_choice = self.parameterAsEnum(parameters, self.GEOMETRY, context)
        use_simplified = self.parameterAsBoolean(
            parameters, self.USE_SIMPLIFIED, context
        )
        join_evidence = self.parameterAsBoolean(parameters, self.JOIN_EVIDENCE, context)

        base_stem = _base_stem(input_file.stem)

        def part(suffix):
            return input_file.parent / f'{base_stem}{suffix}.parquet'

        # The point file carries the canonical attributes itself, so
        # it is both geometry layer and attribute layer; the polygon
        # file holds geometry alone and needs the table joined on.
        if geometry_choice == GEOMETRY_POINTS:
            geo_file = part('_point')
            join_files = []
            if not geo_file.exists():
                feedback.reportError(
                    f'Centroid point file not found: {geo_file.name}. This '
                    'delivery may predate the point file; choose "Boundary '
                    'polygons" instead.'
                )
                return {}
        else:
            geo_file = part('_geo')
            simplified_file = part('_geo_simplified')
            if use_simplified and simplified_file.exists():
                geo_file = simplified_file
                feedback.pushInfo(f'Using simplified geometries: {geo_file.name}')
            elif use_simplified:
                feedback.pushInfo(
                    f'Simplified geometry file not found ({simplified_file.name}), '
                    f'falling back to {geo_file.name}'
                )
            if not geo_file.exists():
                feedback.reportError(f'Geometry file not found: {geo_file}')
                return {}

            attr_file = part('')
            if not attr_file.exists():
                feedback.reportError(f'Attribute file not found: {attr_file}')
                return {}
            join_files = [attr_file]

        if join_evidence:
            evidence_file = part('_evidence')
            if evidence_file.exists():
                join_files.append(evidence_file)
            else:
                feedback.pushInfo(
                    f'No evidence file alongside this delivery '
                    f'({evidence_file.name}); loading without it.'
                )

        layer_name = base_stem
        geo_layer = QgsVectorLayer(str(geo_file), layer_name, 'ogr')
        if not geo_layer.isValid():
            feedback.reportError(f'Failed to load geometry layer: {geo_file}')
            return {}

        join_field = None
        joined_layers = []
        for join_file in join_files:
            join_layer = QgsVectorLayer(
                str(join_file), f'{layer_name}_{join_file.stem}', 'ogr'
            )
            if not join_layer.isValid():
                feedback.reportError(f'Failed to load layer: {join_file}')
                return {}

            field = _resolve_join_field(geo_layer, join_layer)
            if field is None:
                feedback.reportError(
                    f'{geo_file.name} and {join_file.name} share no id column to '
                    "join on (looked for '_join_id', 'geo_id', and any common "
                    'entity id field).'
                )
                return {}
            if join_field is not None and field != join_field:
                feedback.reportError(
                    f'{join_file.name} joins on {field!r} but the delivery is '
                    f'keyed on {join_field!r}; these files are not one set.'
                )
                return {}
            join_field = field

            # The join layer must be in the project for the join to
            # resolve, but stays hidden: its columns surface through
            # the geometry layer.
            QgsProject.instance().addMapLayer(join_layer, False)

            join_info = QgsVectorLayerJoinInfo()
            join_info.setJoinFieldName(join_field)
            join_info.setTargetFieldName(join_field)
            join_info.setJoinLayerId(join_layer.id())
            join_info.setJoinLayer(join_layer)
            join_info.setUsingMemoryCache(True)
            join_info.setPrefix('')

            geo_layer.addJoin(join_info)
            geo_layer.updateFields()
            joined_layers.append(join_layer)

        for join_layer in joined_layers:
            _copy_field_presentation(join_layer, geo_layer, join_field)

        # Keep the join layers referenced so they are not cleaned up.
        geo_layer.setCustomProperty(
            'joined_attr_layer_ids', ','.join(v.id() for v in joined_layers)
        )

        _order_attribute_table(geo_layer, join_field)

        QgsProject.instance().addMapLayer(geo_layer)

        feedback.pushInfo(f'Loaded: {layer_name}')
        feedback.pushInfo(f'  Geometry: {geo_file.name}')
        for join_layer in joined_layers:
            feedback.pushInfo(f'  Joined:   {Path(join_layer.source()).name}')
        feedback.pushInfo(f'  Join key: {join_field}')
        feedback.pushInfo(f'  Features: {geo_layer.featureCount()}')
        feedback.pushInfo(f'  Fields:   {len(geo_layer.fields())}')

        return {'OUTPUT': geo_layer.id()}


def _base_stem(stem):
    """Strip a delivery part suffix off *stem*, if it carries one."""
    for suffix in PART_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_join_field(geo_layer, join_layer):
    """Return the field name to join two delivery layers on, or None.

    '_join_id' and 'geo_id' are the split-layout keys written by
    save_parquet. Everything else falls back to the first field the two
    layers share, which for a delivery is the entity id both files are
    indexed by (e.g. 'footprint_id') -- it leads the file because pandas
    writes the index first.
    """
    geo_fields = [f.name() for f in geo_layer.fields()]
    join_fields = {f.name() for f in join_layer.fields()}

    for preferred in ('_join_id', 'geo_id'):
        if preferred in geo_fields and preferred in join_fields:
            return preferred

    for name in geo_fields:
        if name in join_fields and name not in NON_KEY_FIELDS:
            return name
    return None


def _copy_field_presentation(join_layer, geo_layer, join_field):
    """Carry aliases, editor widgets, and visibility over to the joined layer.

    Categorical values are rendered through their configured labels (Value
    Map / Value Relation widgets), which live on the source layer's field
    configuration and do not follow a join by themselves.
    """
    for join_index, field in enumerate(join_layer.fields()):
        name = field.name()
        if name == join_field:
            continue

        geo_index = geo_layer.fields().indexOf(name)
        if geo_index < 0:
            continue

        alias = join_layer.attributeAlias(join_index)
        if alias:
            geo_layer.setFieldAlias(geo_index, alias)

        setup = join_layer.editorWidgetSetup(join_index)
        if setup and setup.type():
            geo_layer.setEditorWidgetSetup(geo_index, setup)

        flags = join_layer.fieldConfigurationFlags(join_index)
        if flags != geo_layer.fieldConfigurationFlags(geo_index):
            geo_layer.setFieldConfigurationFlags(geo_index, flags)


def _order_attribute_table(geo_layer, join_field):
    """Put the join key first, and hide it when it is a synthetic integer.

    '_join_id' is an internal row number with nothing to say to a reader, so
    it is moved last and hidden. A real entity id ('geo_id', 'footprint_id')
    identifies the row and leads the attribute table instead.
    """
    fields = geo_layer.fields()
    field_names = [f.name() for f in fields]
    if join_field not in field_names:
        return

    key_index = field_names.index(join_field)
    synthetic = join_field == '_join_id'
    others = [i for i in range(len(field_names)) if i != key_index]
    new_order = [*others, key_index] if synthetic else [key_index, *others]

    config = geo_layer.attributeTableConfig()
    config.update(fields)
    columns = config.columns()
    reordered = [columns[i] for i in new_order]

    if synthetic:
        for column in reordered:
            if column.name == join_field:
                column.hidden = True
                break

    config.setColumns(reordered)
    geo_layer.setAttributeTableConfig(config)

    if synthetic:
        # Also keep it out of the Identify Results panel.
        flags = geo_layer.fieldConfigurationFlags(key_index)
        flags |= Qgis.FieldConfigurationFlag.HideFromWms
        geo_layer.setFieldConfigurationFlags(key_index, flags)
