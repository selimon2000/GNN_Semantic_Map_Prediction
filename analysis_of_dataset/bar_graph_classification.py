# import os
# import xml.etree.ElementTree as ET
# from collections import defaultdict

# # Update this to your full path
# base_path = os.path.expanduser('~/capstone_v3/dataset')

# total_spaces = 0
# file_space_counts = []
# space_type_counter = defaultdict(int)

# # Traverse dataset
# for folder in os.listdir(base_path):
#     folder_path = os.path.join(base_path, folder)
#     if not os.path.isdir(folder_path):
#         continue

#     raw_path = os.path.join(folder_path, 'raw')
#     if not os.path.isdir(raw_path):
#         continue

#     for file in os.listdir(raw_path):
#         if file.endswith('.xml'):
#             xml_path = os.path.join(raw_path, file)
#             try:
#                 tree = ET.parse(xml_path)
#                 root = tree.getroot()

#                 spaces = root.findall('space')
#                 num_spaces = len(spaces)
#                 file_space_counts.append(num_spaces)
#                 total_spaces += num_spaces

#                 for space in spaces:
#                     space_type = space.attrib.get('type', '').strip()
#                     if space_type:  # Only count non-empty types
#                         space_type_counter[space_type] += 1

#             except ET.ParseError as e:
#                 print(f"Error parsing {xml_path}: {e}")
#             except Exception as e:
#                 print(f"Unexpected error in {xml_path}: {e}")

# # Print statistics
# print(f"Total XML files processed: {len(file_space_counts)}")
# print(f"Total number of spaces: {total_spaces}")
# print(f"Average number of spaces per file: {total_spaces / len(file_space_counts):.2f}")

# print("\nSpace type counts:")
# for space_type, count in sorted(space_type_counter.items(), key=lambda x: x[1], reverse=True):
#     print(f"{space_type} = {count}")








import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# Update this to your full path
base_path = os.path.expanduser('~/capstone_v3/dataset')

total_spaces = 0
file_space_counts = []
space_type_counter = defaultdict(int)

# Traverse dataset
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(folder_path):
        continue

    raw_path = os.path.join(folder_path, 'raw')
    if not os.path.isdir(raw_path):
        continue

    for file in os.listdir(raw_path):
        if file.endswith('.xml'):
            xml_path = os.path.join(raw_path, file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                spaces = root.findall('space')
                num_spaces = len(spaces)
                file_space_counts.append(num_spaces)
                total_spaces += num_spaces

                for space in spaces:
                    space_type = space.attrib.get('type', '').strip()
                    if not space_type:
                        space_type = 'Other'  # Count empty or missing types as "Other"
                    space_type_counter[space_type] += 1

            except ET.ParseError as e:
                print(f"Error parsing {xml_path}: {e}")
            except Exception as e:
                print(f"Unexpected error in {xml_path}: {e}")

# Print statistics
print(f"Total XML files processed: {len(file_space_counts)}")
print(f"Total number of spaces: {total_spaces}")
print(f"Average number of spaces per file: {total_spaces / len(file_space_counts):.2f}")

print("\nSpace type counts:")
for space_type, count in sorted(space_type_counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{space_type} = {count}")