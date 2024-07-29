from openpyxl import Workbook

# Create a new Workbook object
wb = Workbook()

# Set up the active sheet (default is 'Sheet')
ws = wb.active
ws.title = 'messages'

# Add headers (assuming the structure)
headers = ['ID', 'Name', 'Email', 'Message', 'Timestamp']
ws.append(headers)

# Save the workbook
wb.save('messages.xlsx')

