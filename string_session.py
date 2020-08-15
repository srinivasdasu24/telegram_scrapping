from telethon.sync import TelegramClient
from telethon.sessions import StringSession

api_id = 1348029
api_hash = '3c6e7f9cafaf26130ac6302da5f9614d'
with TelegramClient(StringSession(), api_id, api_hash) as client:
        print(client.session.save())
