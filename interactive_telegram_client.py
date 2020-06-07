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
                msg += "@"+str(user_id)+' ('+self.get_entity(user_id).first_name+(self.get_entity(user_id).last_name if self.get_entity(user_id).last_name is not None else '') + "), "
            else:
                msg +="@"+self.get_entity(user_id).username + ", "
            #msg +="@"+str(user_id)+", "
        msg += " Welcome to the Api testing telegram group"
        self.send_message(self.__default_channel, msg)

    def previous_day_contributors(self):
        info = self.previous_day_group_info()
        for msg in info:
            if msg.message is not None:
                if msg.from_id not in self.group_activity:
                    self.group_activity[msg.from_id]['first_name']=self.get_entity(msg.from_id).first_name
                    self.group_activity[msg.from_id]['last_name']=self.get_entity(msg.from_id).last_name
                    self.group_activity[msg.from_id]['user_name']=self.get_entity(msg.from_id).username
                    self.group_activity[msg.from_id]['messages']=[msg.message]
                    self.group_activity[msg.from_id]['datetime']=[msg.date]
                else:
                    self.group_activity[msg.from_id]['messages'].append(msg.message)
                    self.group_activity[msg.from_id]['datetime'].append(msg.date)
        df=pd.DataFrame(self.group_activity)
        df=df.transpose()
        print(df)
                    



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
            client.previous_day_contributors()
            print("Welcome new members")
            client.welcome_to_group()
            client.disconnect()
        except Exception as e :
            print("Can not Create Client Because : "+str(e))
            print("Please relaunch client !")
