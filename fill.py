import csv
from math import ceil, log10
import pdfrw

ANNOT_KEY = '/Annots'
ANNOT_FIELD_KEY = '/T'
ANNOT_VAL_KEY = '/V'
ANNOT_RECT_KEY = '/Rect'
SUBTYPE_KEY = '/Subtype'
WIDGET_SUBTYPE_KEY = '/Widget'

template = "test0.pdf"

def fill(number):
    with open("./answers/dataset" + number + ".csv", "r") as f:
        reader = csv.reader(f)
        next(reader)

        template_pdf = pdfrw.PdfReader(template)

        for page in template_pdf.pages:
            annotations = page[ANNOT_KEY]
            for annotation in annotations:
                if annotation[SUBTYPE_KEY] == WIDGET_SUBTYPE_KEY:
                    if annotation[ANNOT_FIELD_KEY]:
                        key = annotation[ANNOT_FIELD_KEY][1:-1]
                        # print(key)
                        # print(next(reader))
                        annotation.update(
                                    pdfrw.PdfDict(V='{}'.format(next(reader)[1]))
                                )
                        annotation.update(pdfrw.PdfDict(AP=''))
        template_pdf.Root.AcroForm.update(pdfrw.PdfDict(NeedAppearances=pdfrw.PdfObject('true')))
        pdfrw.PdfWriter().write("./final/answeringsheet" + number + ".pdf", template_pdf)

    
def fill_all(max):
    padding = "%0" + str(ceil(log10(max+1))) + "d"
    for i in range(max + 1):
        fill(padding % (i,))


fill("02")
# fill_all(99)

