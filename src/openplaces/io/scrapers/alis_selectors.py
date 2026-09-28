"""Field names and screen codes for the ALIS registry application.

ALIS is the mainframe-backed search three Massachusetts registries run
instead of masslandrecords: Norfolk (`norfolkresearch.org`, 28 towns
including Brookline), North Essex (`search.lawrencedeeds.com`, 4) and
North Worcester (`fitchburgdeeds.com`, 5). Its field names are
mainframe screen codes rather than anything readable, so they are
named here once with what each one is.

Read off the live Norfolk forms on 2026-09-23; the other two registries
run the same application and are expected to match, which the adapter
checks rather than assumes.
"""

# The application's single entry point. Every screen is this one URL
# with a different `WSIQTP` (which screen) and `WSHTNM` (which form).
ENTRY_PATH = '/ALIS/WW400R.HTM'

# Screen codes, as the registry's own menu links spell them.
#
# `LR01D` is the one worth having. It is titled "Rec Land Name Search",
# but its form carries a town select, a document-type select and a
# from/to date pair, so with the name left out it is the same
# (town, date range, document type) query the Avenu crawler drives.
# `LR09D`, the "Year-Inst" search the maintainer first found, takes a
# year and a single instrument number: crawling with it would mean one
# request per instrument, which is both enormous and rude.
SEARCH_NAME = 'LR01D'
SEARCH_YEAR_INSTRUMENT = 'LR09D'
SEARCH_ALPHA_INDEX = 'LR03D'
SEARCH_PLANS = 'RP01D'
SEARCH_LAND_COURT = 'LC01D'

# Form fields of the name search (screen WW401R00).
FIELD_SURNAME = 'W9SNM'  # 30 chars, blank for a date-only search
FIELD_GIVEN_NAME = 'W9GNM'  # 30 chars
FIELD_DATE_FROM = 'W9FDTA'  # 8 chars, MMDDYYYY
FIELD_DATE_TO = 'W9TDTA'  # 8 chars, MMDDYYYY
FIELD_TOWN = 'W9TOWN'  # select; per-registry codes, e.g. BRKL Brookline
FIELD_DOC_TYPE = 'W9ABR'  # select; 118 options on Norfolk
FIELD_INDEX_TYPE = 'W9IXTP'  # radio: A, R, E
FIELD_RANGE = 'W9INQ'  # radio: AY all years, 1Y, CY current, 50

# Select values that mean "do not narrow on this", spelled with the
# leading asterisk the application uses.
ALL_TOWNS = '*ALL'
ALL_DOC_TYPES = '*ALL'
# The document group that holds deeds. Preferred over enumerating the
# individual deed types, which differ between registries.
DEED_DOC_GROUP = '*DD'

# Hidden fields the form posts with itself. `WSHTNM` names the screen
# the submission belongs to, and the application rejects a request
# whose screen it is not expecting.
FIELD_SCREEN = 'WSHTNM'
FIELD_QUERY_TYPE = 'WSIQTP'
FIELD_KEY_CODE = 'WSKYCD'
FIELD_VERSION = 'WSWVER'

# A submission built as a bare URL, without walking in from the
# registry's own menu, comes back as a 655-byte page titled
# "Land Recs -Error Message" with no message in it (measured on
# Norfolk, 2026-09-23, for a blank name, for a wildcard surname and for
# each range radio). The application is a screen flow that keeps its
# state server side, which is why the adapter navigates rather than
# constructing query strings.
ERROR_TITLE_MARKER = 'error message'
