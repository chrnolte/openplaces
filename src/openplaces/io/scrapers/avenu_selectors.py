"""
CSS/text selectors for the Avenu Insights WebForms registry UI.

Isolated here so DOM changes only require edits in one place.
"""

# Confirmed against live Middlesex South
DATE_FROM = '#SearchFormEx1_DRACSTextBox_DateFrom'
DATE_TO = '#SearchFormEx1_DRACSTextBox_DateTo'
SUBMIT = '#SearchFormEx1_btnSearch'

TOWNS_DROPDOWN = '#SearchFormEx1_ACSDropDownList_Towns'
DOC_TYPE_DROPDOWN = '#SearchFormEx1_ACSDropDownList_DocumentType'

# Results grid — rows alternate DataGridRow / DataGridAlternatingRow
# inside #DocList1_ContentContainer1; header <tr> uses class DataGridHeader
RESULTS_CONTAINER = '#DocList1_ContentContainer1'
RESULTS_TABLE = '#DocList1_GridView_Document'
RESULTS_ROW = (
    '#DocList1_ContentContainer1 tr.DataGridRow, '
    '#DocList1_ContentContainer1 tr.DataGridAlternatingRow'
)

NEXT_BUTTON = '#DocList1_LinkButtonNext'
SCREEN_BLOCKER = '#MessageBoxCtrl1_ScreenBlocker'
MESSAGE_BOX_PANEL = '#MessageBoxCtrl1_UpdatePanel1'
PROGRESS_BAR = '#ProgressBar1_UpdateProgress2'
DOC_LINK = 'input.cssButtonImgSmall16.cssBasketImgButton'

CRITERIA_DATE_SEARCH = '#Navigator1_SearchCriteria1_LinnkButton_15'

# View Details panel (right side, populated when a ButtonRow link is clicked)
DETAILS_CONTAINER = '#DocDetails1_UpdatePanel1'
# ButtonRow elements are <a> tags (not inputs); click the first one per row
DETAILS_ROW_CLICK_TARGET = 'a[id*="ButtonRow"]'

# Incapsula challenge: site requires headless=False
CHALLENGE_FRAME_INDICATOR = 'Incapsula'
REAL_PAGE_INDICATOR = 'D/Default.aspx'

# Transaction-relevant document type display labels (used for dropdown selection).
# Ordered by expected frequency; DEED covers most arm's-length residential sales.
TRANSACTION_DOC_TYPES = [
    'DEED',
    # 'UNITDEED',
    'FORECLOSURE DEED',
    # 'SHERIFFS DEED',
    # 'SHERF DD',
    # 'TREASURERS DEED',
    # 'TREAS DD',
]
