"""

@Author : Dasu Srinivas
@Description : Uses telegram api's to automate telegram tasks

"""
from telethon import TelegramClient,sync
from telethon import functions, types
from telethon.errors import SessionPasswordNeededError
import logging
import datetime
import pytz
import pandas as pd
import configparser
import argparse
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR, Float, Integer, Unicode
from sqlalchemy.dialects import postgresql, mysql
from sqlalchemy import *
from collections import defaultdict

logging.basicConfig(level=logging.ERROR)


class InterActiveTelegramClient(TelegramClient):

    def __init__(self,session,api_id, api_hash, default_channel):

        self.__default_channel = default_channel
        self._id_s=[]
        self.group_activity=defaultdict(dict)

        print('Initializing InteractiveTelegramClient ...')

        super().__init__(
             session=session,api_id=api_id, api_hash=api_hash)

        print('Connecting to Telegram servers...')
        self.start()
        # Ensure you're authorized
        if not self.is_user_authorized():
            self.send_code_request(phone)
            try:
                self.sign_in(phone, input('Enter the code: '))
            except SessionPasswordNeededError:
                self.sign_in(password=input('Password: '))


    def profile(self):
        me = self.get_me()
        print("#########################################")
        print(" \t\tYour Profile:\t\t")
        str = " \tfirstName : {} \n" \
              " \tlastName : {} \n" \
              " \tuserName : {} \n" \
              " \tPhone : {} \n" \
              " \tId : {} \n"
        print(str.format(me.first_name, me.last_name, me.username, me.phone,me.id))
        print("#########################################")

    def load_all_dialogs(self):
        try:

            dialogs = self.get_dialogs(limit=None)

            result = "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format("Number", "Type", "ID", "Username", "Title") + "\n"
            for i, dialog in enumerate(dialogs, start=1):

                entity = dialog.entity
                if type(entity).__name__ == 'Channel':
                    result += "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format(str(i),"Channel/MetaGroup", str(entity.id), str(entity.username),
                                                                str(entity.title)) + "\n"


                elif type(entity).__name__ == "Chat":
                    result += "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format(str(i),"Group", str(entity.id), "-----------",
                                                                    str(entity.title)) + "\n"

                elif type(entity).__name__ == "User":
                    result += "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format(str(i),"User", str(entity.id), str(entity.username),
                                                                    str(entity.first_name)) + "\n"
                else:
                    result += "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format(str(i), "None", str(entity.id),
                                                                           "-----------",
                                                                          str(dialog.name)) + "\n"


            fHandler = open('contacts.txt', 'w', encoding='utf-8')

            fHandler.write(result)
            fHandler.close()

        except Exception as e :
            logger.elog("Telegram Client Can not Fetch All Dialogs Because : " + str(e))
            return None

    def previous_day_group_info(self):
        to_date = pytz.timezone("US/Eastern").localize(datetime.datetime.now()-datetime.timedelta(days=0))
        from_date = pytz.timezone("US/Eastern").localize(datetime.datetime.now() - datetime.timedelta(days=1))

        pre_first_msg = self.get_messages(self.__default_channel, offset_date=from_date, limit=1)[0]
        first_msg = self.get_messages(self.__default_channel, min_id=pre_first_msg.id, limit=1,reverse=True)[0]
        last_msg = self.get_messages(self.__default_channel, offset_date=to_date, limit=1)[0]

        messages_between = self.get_messages(self.__default_channel, min_id=first_msg.id, max_id=last_msg.id) + [first_msg, last_msg]
        return messages_between

    def last_day_joined_users(self):
        messages = self.previous_day_group_info()
        for msg in messages:
            if msg.message is None:
                self._id_s.append(getattr(msg,'from_id'))

    def welcome_to_group(self):
        self.last_day_joined_users()
        msg =''
        for user_id in self._id_s:
            if self.get_entity(user_id).username is None:
                msg += "@"+self.get_entity(user_id).first_name+(self.get_entity(user_id).last_name if self.get_entity(user_id).last_name is not None else '') + ", "
            else:
                msg +="@"+self.get_entity(user_id).username + ", "

        msg += " Welcome to the Co-learning Lounge Telegram channel."
        self.send_message(self.__default_channel, msg)

    def previous_day_contributors(self):
        info = self.previous_day_group_info()
        messages = []
        user_name= []
        user_id = []
        datetime = []
        first_name =[]
        reply_to=[]
        last_name=[]
        message_id=[]
        for msg in info:
            if msg.message is not None:
                if msg.from_id not in self.group_activity:
                    self.group_activity[msg.from_id]['first_name']=self.get_entity(msg.from_id).first_name
                    first_name.append(self.get_entity(msg.from_id).first_name)
                    self.group_activity[msg.from_id]['last_name']=self.get_entity(msg.from_id).last_name
                    last_name.append(self.get_entity(msg.from_id).last_name)
                    self.group_activity[msg.from_id]['user_name']=self.get_entity(msg.from_id).username
                    user_name.append(self.get_entity(msg.from_id).username)
                    self.group_activity[msg.from_id]['messages']=[msg.message]
                    messages.append(msg.message)
                    self.group_activity[msg.from_id]['datetime']=[msg.date]
                    datetime.append(msg.date)
                    message_id.append(msg.id)
                    user_id.append(msg.from_id)
                    reply_to.append(msg.reply_to_msg_id)
                else:
                    self.group_activity[msg.from_id]['messages'].append(msg.message)
                    self.group_activity[msg.from_id]['datetime'].append(msg.date)
                    first_name.append(self.get_entity(msg.from_id).first_name)
                    last_name.append(self.get_entity(msg.from_id).last_name)
                    user_name.append(self.get_entity(msg.from_id).username)
                    messages.append(msg.message)
                    datetime.append(msg.date)
                    user_id.append(msg.from_id)
                    message_id.append(msg.id)
                    reply_to.append(msg.reply_to_msg_id)

        for i in range(len(messages)):
            if messages[i] is '':
                messages[i]= ','
        data ={'first_name' :first_name, 'last_name':last_name, 'user_name':user_name,'user_id':user_id, 'message':messages,'timestamp':datetime,'reply_to_msg_id':reply_to,'message_id':message_id}
        tele_df = pd.DataFrame(data)
        tele_df['date'] = [d.date() for d in tele_df['timestamp']]
        tele_df['time'] = [d.time() for d in tele_df['timestamp']]
        #print(tele_df)
        df=pd.DataFrame(self.group_activity)
        df=df.transpose()
        #df['messages'] = df.messages.apply(lambda x: '##'.join([str(i) for i in x]))
        #print(type(df['messages']))
        return tele_df

    def mapping_df_types(self,df):
        dtypedict = {}
        for i, j in zip(df.columns, df.dtypes):
            if "object" in str(j) and "message" in str(i): 
                dtypedict.update({i: NVARCHAR(length=12000)})
            else:
                dtypedict.update({i: NVARCHAR(length=255)})
            if "float" in str(j):
                dtypedict.update({i: Float(precision=2, asdecimal=True)})
            if "int" in str(j):
                dtypedict.update({i: Integer()})
        return dtypedict

    def save_to_sql(self,df):
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="telegram_user",
                               pw="tele123",
                               db="telegram"),pool_pre_ping=True)
        df.to_sql(name='tele_chat', con = engine, if_exists = 'append', index=False,dtype=self.mapping_df_types(df))
                    



if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-id", "--api_id", dest="api_id",type=int,
            help="It's a mandatory, Need to give appication id, example id is : 4561235",required=True,default=None)
    arg_parse.add_argument("-hash", "--api_hash", dest="api_hash",type=str,
                           help="It's a mandatory argument, need to give application hash id ", required=True,default=None)
    arg_parse.add_argument("-s", "--session", dest="session",type=str,
                           help="It's a optional argument", default=None,required=False)
    arg_parse.add_argument("-dc", "--channel", dest="channel",type=str,
                           help="Please provide channel name on which you want to perform this operation", default="default_channel")
    args = arg_parse.parse_args()
    if args.api_id is not None or args.api_hash is not None:
        #api_id = args.api_id
        #api_hash = '3c6e7f9cafaf26130ac6302da5f9614d'
        #session = 'srinivas dasu.session'
        #default_channel = 'ColearningLounge_AIRoom'
        try:
            client = InterActiveTelegramClient(args.session,args.api_id, args.api_hash,args.channel)
            client.profile()
            print("Previous Day contributors ...")
            client.save_to_sql(client.previous_day_contributors())
            print("Welcome new members")
            client.welcome_to_group()
            client.disconnect()
        except Exception as e :
            print("Can not Create Client Because : "+str(e))
            print("Please relaunch client !")
