import re
import os
import subprocess
import hashlib

# Input and output paths
txt_path = 'the_sound_and_the_fury.txt'  # English original text
output_dir = 'corpus'
os.makedirs(output_dir, exist_ok=True)


# Read the full text
def read_text(path):
    with open(path, 'r', encoding='utf-8') as file_handle:
        return file_handle.read()


# Split into sections
def split_sections(raw_text):
    # Pattern matches section headings (case-insensitive)
    pattern = re.compile(r"\b(APRIL SEVENTH, 1928|JUNE SECOND, 1910|APRIL SIXTH, 1928|APRIL EIGHTH, 1928)\b",
                         re.IGNORECASE)
    parts = pattern.split(raw_text)
    split_result = []
    # parts: [..., heading, content, heading, content, ...]
    for i in range(1, len(parts), 2):
        section_title = parts[i].strip().capitalize()
        section_content = parts[i + 1].strip()
        split_result.append((section_title, section_content))
    return split_result


# Wrap content in TEI-lite
def wrap_tei(section_title, section_content):
    # Split paragraphs by blank lines
    paras = re.split(r"\n\s*\n", section_content)
    # Build TEI document
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<TEI xmlns="http://www.tei-c.org/ns/1.0">', '  <text>',
             '    <body>', f'      <div type="section" xml:id="{section_title}">']
    # Add paragraphs
    for p in paras:
        p = p.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        lines.append(f'        <p>{p.strip()}</p>')
    lines.append('      </div>')
    lines.append('    </body>')
    lines.append('  </text>')
    lines.append('</TEI>')
    return '\n'.join(lines)


# Validate XML using xmllint (requires it to be installed)
def validate_xml(file_path):
    try:
        subprocess.run(['xmllint', '--noout', file_path], check=True)
        print(f'Validated XML: {file_path}')
    except subprocess.CalledProcessError:
        print(f'Validation failed: {file_path}')


# Main execution
if __name__ == '__main__':
    text = read_text(txt_path)
    sections = split_sections(text)
    for section_title, section_content in sections:
        tei_xml = wrap_tei(section_title, section_content)
        out_path = os.path.join(output_dir, f'{section_title}.xml')
        with open(out_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write(tei_xml)
        print(f'Written section: {out_path}')
