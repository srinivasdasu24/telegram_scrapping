from telethon import TelegramClient, sync
import pandas as pd
import datetime

api_id = 1348029
api_hash = '3c6e7f9cafaf26130ac6302da5f9614d'
group_username = 'C0ban_global' # Group name can be found in group link (Example group link : https://t.me/c0ban_global, group name = 'c0ban_global')
client = TelegramClient('srinivas dasu', api_id, api_hash).start()
participants = client.get_participants(group_username)
#print(participants.stringify())
firstname =[]
lastname = []
username = []
id = []
if len(participants):
    for x in participants:
        firstname.append(x.first_name)
        lastname.append(x.last_name)
        username.append(x.username)
        id.append(x.id)

# list to data frame conversion

data ={'first_name' :firstname, 'last_name':lastname, 'user_name':username, 'id':id}

userdetails = pd.DataFrame(data)
#print(userdetails)

chats =client.get_messages(group_username, 15) # n number of messages to be extracted
# Get message id, message, sender id, reply to message id, and timestamp
message_id =[]
message =[]
sender =[]
reply_to =[]
time = []
if len(chats):
    for chat in chats:
        message_id.append(chat.id)
        message.append(chat.message)
        sender.append(chat.from_id)
        reply_to.append(chat.reply_to_msg_id)
        time.append(chat.date)
data ={'message_id':message_id, 'message': message, 'id':sender, 'reply_to_msg_id':reply_to, 'time':time}
df = pd.DataFrame(data)
condit_data=pd.merge(userdetails,df,on='id')
print(condit_data)

print("------- message deatils-------")
print(df)

messages =[]
time = []
destination_user_username='yogesh'
entity=client.get_entity(destination_user_username)
for message in client.iter_messages(group_username, search='c0ban',reverse=True,offset_date=datetime.datetime(2020,1,10),entity=entity):
    messages.append(message.message) # get messages
    time.append(message.date) # get timestamp
data ={'time':time, 'message':messages}

df = pd.DataFrame(data)
print("------------Messages with specific keyword(here keyword: 'c0ban')--------")
print(df)
