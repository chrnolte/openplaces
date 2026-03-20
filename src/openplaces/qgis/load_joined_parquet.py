"""
QGIS processing algorithm (plugin) to load joined
Parquet file from openplaces filesystem ('*.parquet'
file and geometries in associated '*_geo.parquet',
linked by '_join_id' or 'geo_id'
"""

from pathlib import Path

from qgis.core import (
    Qgis,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProject,
    QgsVectorLayer,
    QgsVectorLayerJoinInfo,
)


class LoadJoinedParquetAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'

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
            'Load attribute and geometry parquet files with automatic join on _join_id'
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT, 'Attribute or Geometry Parquet File', extension='parquet'
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        input_file = Path(self.parameterAsFile(parameters, self.INPUT, context))

        # Determine which file was dropped
        if input_file.stem.endswith('_geo'):
            geo_file = input_file
            attr_file = input_file.parent / input_file.name.replace(
                '_geo.parquet', '.parquet'
            )
        else:
            attr_file = input_file
            geo_file = input_file.parent / f'{input_file.stem}_geo.parquet'

        if not geo_file.exists():
            feedback.reportError(f'Geometry file not found: {geo_file}')
            return {}

        if not attr_file.exists():
            feedback.reportError(f'Attribute file not found: {attr_file}')
            return {}

        layer_name = attr_file.stem

        # Load GEOMETRY layer as the base (has geometry, so it becomes a vector layer)
        geo_layer = QgsVectorLayer(str(geo_file), layer_name, 'ogr')

        if not geo_layer.isValid():
            feedback.reportError(f'Failed to load geometry layer: {geo_file}')
            return {}

        # Load attribute layer as the join source (hidden)
        attr_layer = QgsVectorLayer(str(attr_file), f'{layer_name}_attr', 'ogr')

        if not attr_layer.isValid():
            feedback.reportError(f'Failed to load attribute layer: {attr_file}')
            return {}

        # Determine join field from geometry layer
        geo_field_names = [f.name() for f in geo_layer.fields()]
        if '_join_id' in geo_field_names:
            join_field = '_join_id'
        elif 'geo_id' in geo_field_names:
            join_field = 'geo_id'
        else:
            feedback.reportError(
                "Neither '_join_id' nor 'geo_id' found in geometry file"
            )
            return {}

        # Verify join field exists in attribute layer
        attr_field_names = [f.name() for f in attr_layer.fields()]
        if join_field not in attr_field_names:
            feedback.reportError(
                f"Join field '{join_field}' not found in attribute file"
            )
            return {}

        # Add attribute layer to project hidden (needed for join to work)
        QgsProject.instance().addMapLayer(attr_layer, False)

        # Configure join: attr_layer attributes are joined onto geo_layer
        join_info = QgsVectorLayerJoinInfo()
        join_info.setJoinFieldName(join_field)
        join_info.setTargetFieldName(join_field)
        join_info.setJoinLayerId(attr_layer.id())
        join_info.setJoinLayer(attr_layer)
        join_info.setUsingMemoryCache(True)
        join_info.setPrefix('')

        geo_layer.addJoin(join_info)

        # Keep attribute layer reference to prevent cleanup issues
        geo_layer.setCustomProperty('joined_attr_layer_id', attr_layer.id())

        # Reorder fields: index first, join_field last
        fields = geo_layer.fields()
        field_names = [f.name() for f in fields]

        # Index field: first field from attribute layer (was the DataFrame index)
        index_field_name = attr_field_names[0] if attr_field_names else None

        # Find positions
        join_id_idx = next(
            (i for i, name in enumerate(field_names) if name == join_field), None
        )
        index_idx = next(
            (i for i, name in enumerate(field_names) if name == index_field_name), None
        )

        # Create new field order: index first, all others, join_field last
        new_order = []

        if index_idx is not None:
            new_order.append(index_idx)

        for i in range(len(field_names)):
            if i != index_idx and i != join_id_idx:
                new_order.append(i)

        if join_id_idx is not None:
            new_order.append(join_id_idx)

        # Apply attribute table reordering
        config = geo_layer.attributeTableConfig()
        config.update(fields)
        columns = config.columns()
        reordered_columns = [columns[i] for i in new_order]

        # Hide join_field column from attribute table
        if join_id_idx is not None:
            for col in reordered_columns:
                if col.name == join_field:
                    col.hidden = True
                    break

        config.setColumns(reordered_columns)
        geo_layer.setAttributeTableConfig(config)

        # Hide join_field from Identify Results
        if join_id_idx is not None:
            flags = geo_layer.fieldConfigurationFlags(join_id_idx)
            flags |= Qgis.FieldConfigurationFlag.HideFromWms
            geo_layer.setFieldConfigurationFlags(join_id_idx, flags)

        # Add geometry layer (with joined attributes) to project
        QgsProject.instance().addMapLayer(geo_layer)

        feedback.pushInfo(f'Loaded: {layer_name}')
        feedback.pushInfo(f'  Features: {geo_layer.featureCount()}')
        feedback.pushInfo(f'  Fields: {len(geo_layer.fields())}')

        return {'OUTPUT': geo_layer.id()}
