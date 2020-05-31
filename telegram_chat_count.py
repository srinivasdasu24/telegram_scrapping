

from telethon.tl.types import User
from telethon import TelegramClient,sync
from telethon.errors import SessionPasswordNeededError

# (1) Use your own values here
api_id = api_id # eg: 17349
api_hash = 'api_hash' #eg :'344583e45741c457fe1862106095a5eb'

phone = 'your_phone_number' # '919603594976'
username = 'your_user_name' #eg 'dasu24'

# (2) Create the client and connect
client = TelegramClient(username, api_id, api_hash)
client.connect()

# (3) Ensure you're authorized
if not client.is_user_authorized():
    client.send_code_request(phone)
    try:
        client.sign_in(phone, input('Enter the code: '))
    except SessionPasswordNeededError:
        client.sign_in(password=input('Password: '))

me = client.get_me()
print(me)

# (4) Get the message count from all groups where 'user_name' is a member and individaul chats 
entities = client.get_dialogs(limit=30)

counts = []
for e in entities:
    if isinstance(e, User):
        name = e.first_name
    else:
        name = e.title

    count= client.get_messages(e, limit=1)
    counts.append((name, count.total))
# (5) Sort the group name and count based on count 
counts.sort(key=lambda x: x[1], reverse=True)
for name, count in counts:
    print('{}: {}'.format(name, count))
