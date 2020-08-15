import json
from telethon import TelegramClient, sync
import pytz
from collections import OrderedDict
from telethon.sessions import StringSession
import datetime

string = '1BVtsOKQBu1ik3MO5LP5u_1COFNK8Su7pU51bcg1a79Uqivyx0r0CZuCuqgJrfiESsm-0UhUCzFFQAIfisrGBsEjnXwM3rGHGyMYUYHo8Xv4BURY6YhxKCCpeXRebnTwCjAvVx2SKSwW1kVuHJN5080-7Lt2HqPjuOrdlJt-ch-K9-BmNt1E-K6iVG8l-ekPuvGBYANwLjTaOTgDSLrPW5uR5qs9R_iSeAGzISIQ3FYoWYNeW5N_eqJ38GapyiEp_tADeQc5HjD7GwO9iI_-BAHAyytmALr6moXcUx1Bho-wV4lTza59WFa9dJ6sMNa3VYKK6jlSvSgY6r_0OiQHOOWQ3Htj0J1o='
_id_s=[]
def previous_day_group_info(client,__default_channel):
    to_date = pytz.timezone("US/Eastern").localize(datetime.datetime.now().replace(hour=14,minute=30,second=0, microsecond=0)-datetime.timedelta(days=1))
    from_date = pytz.timezone("US/Eastern").localize(datetime.datetime.now().replace(hour=14,minute=30,second=0, microsecond=0) - datetime.timedelta(days=2))

    pre_first_msg = client.get_messages(__default_channel, offset_date=from_date, limit=1)[0]
    first_msg = client.get_messages(__default_channel, min_id=pre_first_msg.id, limit=1,reverse=True)[0]
    last_msg = client.get_messages(__default_channel, offset_date=to_date, limit=1)[0]
    print(first_msg)
    print(last_msg)
    messages_between = client.get_messages(__default_channel, min_id=first_msg.id, max_id=last_msg.id) + [first_msg, last_msg]
    return messages_between
                          
def last_day_joined_users(messages):
    for msg in messages:
        if msg.message is None and msg.reply_to_msg_id is None:
            if msg.action.users[0] == msg.from_id:
                _id_s.append(getattr(msg,'from_id'))
            else:
                _id_s.append(getattr(msg,'action').users[0])

def welcome_to_group(client,group_username):
    msg =''
    for user_id in _id_s:
        if client.get_entity(user_id).username is None:
            #msg += "@"+client.get_entity(user_id).first_name+(client.get_entity(user_id).last_name if client.get_entity(user_id).last_name is not None else '') + ", "
            msg += "["+client.get_entity(user_id).first_name+(client.get_entity(user_id).last_name if client.get_entity(user_id).last_name is not None else '') +"](tg://user?id="+str(user_id)+")"+", "
        else:
            msg +="@"+client.get_entity(user_id).username + ", "
    print(msg)
    msg = msg[:-2]
    msg += "\nWelcome to the Co-learning Lounge community where we are transforming education by open-sourcing through community collaboration. Please briefly introduce yourself. Also, share your Linkedin, Medium(if any) profile so others can follow and connect with you."
    msg += " Also, Please join, subscribe and follow our other social media platforms to never miss any update: [co-learning lounge](https://linktr.ee/colearninglounge)"
    client.send_message(group_username, msg, parse_mode="Markdown")

def lambda_handler(event, context):
    api_id = 1348029
    api_hash = '3c6e7f9cafaf26130ac6302da5f9614d'
    group_username = 'ColearningLounge_AIRoom'
    participant_ids={}
    client = TelegramClient(StringSession(string), api_id, api_hash).start()
    participants = client.get_participants(group_username)
    for user in participants:
        participant_ids[user.id]=False
    participant_ids=OrderedDict(sorted(participant_ids.items(), key=lambda t: t[0]))
    msgs = previous_day_group_info(client,group_username)
    last_day_joined_users(msgs)
    welcome_to_group(client,group_username)

lambda_handler("hlo","hi")
