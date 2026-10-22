"""`read_members`: several extracted files read as one table."""

from openplaces.io.ingester.table_read import _TableReadMixin

_XML = (
    '<?xml version="1.0"?><Extract><List><formData><formHeader>'
    '<returnID>{rid}</returnID><span>{span}</span>'
    '</formHeader></formData></List></Extract>'
)


class _Reader(_TableReadMixin):
    def __init__(self, glob):
        self.recipe = {'xml_xpath': './/formHeader'}
        self.download_partition = {'read_members_glob': glob}


def test_every_matching_member_is_read_in_name_order(tmp_path):
    # Fabricated ids; a monthly and an overlapping weekly extract.
    (tmp_path / 'PT20230101_20230131.xml').write_text(
        _XML.format(rid='A1', span='00000000001')
    )
    (tmp_path / 'PT20230102_20230108.xml').write_text(
        _XML.format(rid='A2', span='00000000002')
    )
    (tmp_path / 'PT20220101_20221231.xml').write_text(
        _XML.format(rid='B1', span='00000000003')
    )
    reader = _Reader('PT2023*.xml')
    df = reader._read_flat_table(tmp_path / 'PT20230101_20230131.xml', None, None)
    assert df['returnID'].tolist() == ['A1', 'A2']
    # Text, so a SPAN keeps its leading zeros.
    assert df['span'].tolist() == ['00000000001', '00000000002']


def test_without_read_members_only_the_file_is_read(tmp_path):
    path = tmp_path / 'PT20230101_20230131.xml'
    path.write_text(_XML.format(rid='A1', span='00000000001'))
    (tmp_path / 'PT20230102_20230108.xml').write_text(
        _XML.format(rid='A2', span='00000000002')
    )
    reader = _Reader(None)
    assert reader._read_flat_table(path, None, None)['returnID'].tolist() == ['A1']
