import xml.etree.ElementTree as ET
import json
import sys

shared_strings_file = 'xl/sharedStrings.xml'
sheet_file = 'xl/worksheets/sheet1.xml'

ns = {'ns': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

tree = ET.parse(shared_strings_file)
root = tree.getroot()
strings = []
for si in root.findall('ns:si', ns):
    t_elem = si.find('ns:t', ns)
    if t_elem is not None:
        text = t_elem.text if t_elem.text is not None else ''
        strings.append(text)
    else:
        strings.append('')

print(f"Total shared strings: {len(strings)}")

tree2 = ET.parse(sheet_file)
root2 = tree2.getroot()
sheet_data = root2.find('ns:sheetData', ns)
if sheet_data is None:
    print("sheetData not found")
    sys.exit(1)

rows = sheet_data.findall('ns:row', ns)
print(f"Total rows in sheet: {len(rows)}")

data = []
for row in rows:
    row_num = row.get('r')
    cells = row.findall('ns:c', ns)
    row_dict = {}
    for cell in cells:
        cell_ref = cell.get('r')
        cell_type = cell.get('t')
        value_elem = cell.find('ns:v', ns)
        value = ''
        if value_elem is not None:
            value = value_elem.text
            if cell_type == 's':
                try:
                    idx = int(value)
                    value = strings[idx] if idx < len(strings) else ''
                except ValueError:
                    pass
        row_dict[cell_ref] = value
    data.append((row_num, row_dict))

header_row = data[0][1] if data else None
if header_row:
    print("Header detected")
    column_map = {}
    for cell_ref, value in header_row.items():
        column_map[cell_ref[0]] = value
    print("Column mapping:", column_map)

output = []
for i in range(1, len(data)):
    row_num, row_dict = data[i]
    item = {}
    for cell_ref, value in row_dict.items():
        col = cell_ref[0]
        if col in 'ABCDEF':
            header = column_map.get(col, col)
            item[header] = value
    output.append(item)

print(f"Total issues extracted: {len(output)}")

with open('/tmp/issues_parsed.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("Data saved to /tmp/issues_parsed.json")

with open('/tmp/issues_sample.txt', 'w', encoding='utf-8') as f:
    for i, item in enumerate(output[:5]):
        f.write(f"Issue #{i+1}:\n")
        for key, value in item.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
print("Sample saved to /tmp/issues_sample.txt")