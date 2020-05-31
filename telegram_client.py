from telethon import TelegramClient, sync
import pandas as pd
import datetime

# (1) Use your own values here 
api_id = 'your_api_id' # api_id
api_hash = 'your_api_hash' #'api_hash'
group_username = 'C0ban_global' # Group name can be found in group link (Example group link : https://t.me/c0ban_global, group name = 'c0ban_global')
# (2) creating telegram clinet and start the client
client = TelegramClient('srinivas dasu', api_id, api_hash).start()

# (3) Get the members of the group you are interested and intrested parameters

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

# (4) Get messages from group and you can limit the number of messages, below we are limiting to 15

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

# (5) Perofming join operation on two data frames to get the combined data from 2 frames
condit_data=pd.merge(userdetails,df,on='id') # merging the two data frames based on common column
#condit_data=pd.merge(userdetails,df,left_on='id',right_on='sender_Id') # if two dataframes have different column names
print(condit_data)


# (6) Seraching the messages in the group with the 'keyword' we are interested 
print("------- message deatils-------")
print(df)

messages =[]
time = []
destination_user_username='user_name_you_want_to_fetch_messages'
entity=client.get_entity(destination_user_username)

#revrse can be used when entity or id is not empty, so we need to have user_id or user_name to use reverse

for message in client.iter_messages(group_username, search='c0ban',reverse=True,offset_date=datetime.datetime(2020,1,10),entity=entity): 
    messages.append(message.message) # get messages
    time.append(message.date) # get timestamp
data ={'time':time, 'message':messages}

df = pd.DataFrame(data)
print("------------Messages with specific keyword(here keyword: 'c0ban')--------")
print(df)
