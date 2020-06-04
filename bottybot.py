#!/usr/bin/python3
import telegram
import time
import requests
import os
from random import randint
import threading
from sys import platform
from shutil import copyfile

class StylingFred(threading.Thread):
    def __init__(self,token,chat_id, content_image, style_image,target_folder,its=250):
        threading.Thread.__init__(self)
        self.token = token
        self.cache = content_image
        self.style_image = style_image
        self.folder = target_folder
        self.iterations = its
        self.chat_id = chat_id
        self.done = False

    def run(self):
        self.work()
        self.done= True

    def send_image(self, img_pfad, caption = 'Do you like it? You can /set_in_cache and /dream it!'):
        while True:
            try:
                url = "https://api.telegram.org/bot" + self.token + "/sendPhoto"
                files = {'photo': open(img_pfad, 'rb')}
                data = {'chat_id': self.chat_id, 'caption':caption}
                requests.post(url, files=files, data=data)
                return
            except:
                print("Try again")
                time.sleep(5)

    def work(self):
        if platform == 'win64':
          os.system("python NN/StyleClass.py --input_image " + self.cache + " --target_folder "
                    + self.folder +"StyledImages/" + " --iterations " + str(self.iterations)
                    + "  --style_image " + self.style_image)
        else:
          os.system("python3 NN/StyleClass.py --input_image " + self.cache + " --target_folder "
                    + self.folder +"StyledImages/" + " --iterations " + str(self.iterations)
                    + "  --style_image " + self.style_image)
        pics = os.listdir(self.folder+"StyledImages")
        pics.sort()
        self.send_image(self.folder+"StyledImages/"+pics[-1])

class DreamingFred(StylingFred):
    def __init__(self, token, chat_id, content_image, target_folder, iterations, layer):
        threading.Thread.__init__(self)
        self.token = token
        self.cache = content_image
        self.folder = target_folder
        self.iterations = iterations
        self.layer = layer
        self.chat_id = chat_id
        self.done = False

    def work(self):
        if "win" in platform:
          os.system("python NN/DDream.py --input_image " + self.cache + " --target_folder "
                    + self.folder +"StyledImages/" + " --iterations " + str(self.iterations)
                    + " --at_layer " + str(self.layer))
        else:
          os.system("python3 NN/DDream.py --input_image " + self.cache + " --target_folder "
                    + self.folder +"StyledImages/" + " --iterations " + str(self.iterations)
                    + " --at_layer " + str(self.layer))
        pics = os.listdir(self.folder+"StyledImages")
        pics.sort()
        self.send_image(self.folder+"StyledImages/"+pics[-1],"Do you like it? You can /set_in_cache and /style it")

class EinChat():
    def __init__(self, Bot, chat_id):
        self.Bot = Bot
        self.chat_id = int(chat_id)
        if not os.path.exists("chats/"+str(self.chat_id)):
           os.mkdir("chats/"+str(self.chat_id))
           os.mkdir("chats/"+str(self.chat_id)+"/Images")
           os.mkdir("chats/"+str(self.chat_id)+"/StyledImages")
        self.event = 0
        self.user_events = []
        self.folder = "chats/"+str(self.chat_id)+"/"
        self.style_id=999
        self.next_style()
        self.all_styles = False
        self.cache = ''
        self.options = {"all":lambda:self.all(),#admin option
                        "/dream":lambda:self.dream(),
                        "delete":lambda:self.delstyle(),#admin option
                        "/rdream":lambda:self.dream(randint(20,40),randint(15,35)),
                        "load":lambda:self.load_cache(),
                        "offline":lambda:self.set_offline(),#admin option
                        "online":lambda:self.set_online(),#admin option
                        "n":lambda:self.next_style(True),
                        "/next":lambda:self.next_style(True),
                        "/set_in_cache":lambda:self.set_cache(),
			            "/start":lambda:self.send_message("Heylo, welcome to my image style dreaming transfer bot =)."
                                                          " Check it out by sending a picture to me!"),
                        "showcache":lambda:self.send_image(self.cache, self.cache[self.cache.find("Images/")+7:]),
                        "showstyle":lambda:self.send_image(self.style_image,
                                                           self.style_image[self.style_image.find("[")
                                                           + 1:self.style_image.find("]")]),
                        "/style":lambda:self.style(),
                        "/s":lambda:self.style(),
                        "turnoff":lambda:self.turnoff()#admin option
                        }

    def next_style(self, capt=False):
        if len(os.listdir("Styles/"))>self.style_id+1:
           self.style_id += 1
        else:
           self.style_id = 0
        styles = os.listdir("Styles/")
        styles.sort()
        if capt:
          self.send_message("New /style is " + styles[self.style_id]+" or take the /next one?")
        self.style_image = "Styles/"+styles[self.style_id]

    def set_cache(self):
        pics = os.listdir(self.folder+"StyledImages")
        pics.sort()
        self.cache = pics[-1]
        self.send_message("Last recieved picture set in cache!")
        self.cache = self.folder + "StyledImages/"+self.cache

    def dream(self, its = 25, lay = 27):
        if not len(self.cache):
            self.send_message("You have to send a picture first, pls!")
            return
        if self.Bot.online:
            self.send_message("Do androids dream about electric sheep...?\n"
                              + (len(self.Bot.AlleFreds)>0)*(str(len(self.Bot.AlleFreds))+" in queue"))
            print("Dreaming with",its,"iterationen in layer",lay)
            self.Bot.AlleFreds.append(DreamingFred(self.Bot.token,self.chat_id,self.cache,self.folder,its,lay))
        else:
            self.send_message("Sorry, the service is atm unavailable.\n")

    def style(self,its = 250):
        if not len(self.cache):
            self.send_message("You have to send a picture first, pls!")
            return
        if self.Bot.online:
            self.send_message("Do a style transfer with " +self.style_image[7:-4] +".\n"
                              + (len(self.Bot.AlleFreds)>0)*(str(len(self.Bot.AlleFreds))+" in queue. ")
                              + "Go to /next style?")
            self.Bot.AlleFreds.append(StylingFred(self.Bot.token, self.chat_id,
                                                  self.cache, self.style_image, self.folder,its))
        else:
            self.send_message("Sorry, the service is atm unavailable.\n")

    def load_cache(self):
        bilder = os.listdir(self.folder+"Images")
        if len(bilder):
           bilder.sort()
           self.cache = bilder[-1]
           self.send_message(self.cache + " loaded")
           self.cache = self.folder+"Images/"+self.cache
        else:
           self.send_message("Your cache is empty!")
    
    def download_pic(self,file_id):
        URL = self.Bot.bot.getFile(file_id)['file_path']
        image_number = 0
        dapath = self.folder+ "Images/"
        while os.path.exists((dapath+"Image"+(4-len(str(image_number)))*"0"+str(image_number)+".jpg")):
           image_number += 1
        img_data = requests.get(URL).content
        with open(dapath+"Image"+(4-len(str(image_number)))*"0"+str(image_number)+".jpg","wb") as handler:
            handler.write(img_data)
        self.cache = dapath+"Image"+(4-len(str(image_number)))*"0"+str(image_number)+".jpg"

    def send_image(self, img_pfad, caption = ''):
        self.Bot.bot.send_chat_action(self.chat_id,action=telegram.ChatAction.UPLOAD_PHOTO)
        while True:
            try:
                url = "https://api.telegram.org/bot" + self.Bot.MeinBot + "/sendPhoto"
                files = {'photo': open(img_pfad, 'rb')}
                data = {'chat_id': self.chat_id, 'caption':caption}
                requests.post(url, files=files, data=data)
                return
            except:
                print("Try again")
                time.sleep(5)

    def send_message(self,message):
        while True:
            try:
                self.Bot.bot.send_chat_action(self.chat_id, action=telegram.ChatAction.TYPING)
                time.sleep(0.1)
                self.Bot.bot.send_message(self.chat_id, message)
                return
            except:
                print("Try again")
                time.sleep(1)

    def run_events(self):
      if len(self.user_events):
        self.event = self.user_events.pop(0)
        try:
          if self.event[0] == "Message":
            print(self.chat_id,"schrieb",self.event[1])
            self.options[self.event[1].lower()]()
          elif self.event[0] == "Photo":
            print(self.chat_id,"sent a pic")
            self.download_pic(self.event[1])
            self.send_message("Thx for the pic <3.\nYou want to /style or /dream about it?\n/rdream")
        except:
          try:
            self.admin()
          except:
            self.send_message("Unknown command")

class Admin(EinChat):
    def __init__(self, Bot, chat_id):
        EinChat.__init__(self,Bot,chat_id)
        self.send_message("Du bist ein Admin")
    def set_offline(self):
        self.Bot.online=False
        self.send_message("Styling turned off")

    def set_online(self):
        self.Bot.online =True
        self.send_message("Styling turned on")

    def delstyle(self):
        styles = os.listdir("Styles/")
        styles.sort()
        os.remove("Styles/"+styles[self.style_id])

    def all(self):
        for i in range(len(os.listdir("Styles/"))):
          self.next_style()
          self.style()

    def turnoff(self):
        self.send_message("Bot is shutting down")
        self.Bot.running=False

    def admin(self):
        if not self.event[1].lower().find("setstyle"):
            pics = os.listdir(self.folder+"Images")
            if len(pics):
                pics.sort()
                newstyle = pics[-1]
                copyfile(self.folder+"Images/"+newstyle,"Styles/["+self.event[1].lower()[9:]+"].jpg")
        elif not self.event[1].lower().find("toall:"):
            self.Bot.send_to_all(self.event[1][6:])
        else:
            self.send_message("Unknown command")

class StylingBot():
  online=True
  def __init__(self):
     self.token = ''
     with open("token.txt") as f:
        for line in f:
            self.token = line[:-1]
     self.bot  =  telegram.Bot(token=self.token)
     self.alle_chats = []
     self.chat_ids = []
     self.AlleFreds = []
     self.lade_chats()
     self.event_list = []
     self.off = 0
     self.running=True
     self.mainloop()

  def lade_chats(self):
      if not os.path.exists("chats/"):
          os.mkdir("chats/")
      with open("admin.txt") as f:
          for line in f:
              print(line[:-1],"is an admin")
              self.chat_ids.append(int(line))
              self.alle_chats.append(Admin(self,line[:-1]))
      liste = os.listdir("chats/")
      for line in liste:
         chat_id = int(line)
         if not chat_id in self.chat_ids:
            print(chat_id)
            self.chat_ids.append(chat_id)
            self.alle_chats.append(EinChat(self,chat_id))

  def send_to_all(self, message):
      for chat in self.alle_chats:
        chat.send_message(message)

  def get_event_list(self):
      self.event_list = []
      try:
          self.event_list = self.bot.getUpdates(offset=self.off)
      except:
          print("No connection to server")
          time.sleep(5)

  def check_new_chats(self):
      for event in self.event_list:
          chat_id = int(event['message']['chat']['id'])
          if not chat_id in self.chat_ids:
              self.chat_ids.append(chat_id)
              self.alle_chats.append(EinChat(self, chat_id))

  def mainloop(self):
     while self.running:
        self.get_event_list()
        self.check_new_chats()
        for event in self.event_list:
            for chat in self.alle_chats:
                if chat.chat_id == event['message']['chat']['id']:
                    if event['message']['text']:
                        chat.user_events.append(["Message",event['message']['text']])
                    elif event['message']['photo']:
                        chat.user_events.append(["Photo",event['message']['photo'][-1]['file_id'],
                                                 event['message']['caption']])
                    else:
                        print(event['message'])
                    self.off = event['update_id']+1
        for chat in self.alle_chats:
            chat.run_events()
        if len(self.AlleFreds) and self.online:
           if self.AlleFreds[0].done:
               if len(self.AlleFreds)>1:
                   self.AlleFreds = self.AlleFreds[1:]
                   continue
               else:
                   self.AlleFreds = []
                   continue
           if not self.AlleFreds[0].isAlive():
               self.AlleFreds[0].start()
        time.sleep(1)
        self.get_event_list()

if __name__ == '__main__':
    StylingBot()