from datetime import datetime
from tkinter import *
import textwrap
import customtkinter




class ChatBotEngine:
    """Chat Bot Engine Interface"""

    def __init__(self):
        pass

    def user_entry(self, text: str) -> str:
        """Example Chat Bot User API"""

        
        text = text

        return "BOT: " + "Sample response from Mock Chatbox API"


class ChatBotApp(customtkinter.CTk):
    """Chat Bot Application HMI"""

    def __init__(self, chatbot: ChatBotEngine) -> None:
        """Constructor"""
        super().__init__()

        self.hint_flag = 0      
        self.engine = chatbot   

        self.title("QuinSpark Chat Bot")
        self.geometry("400x500")
        self.resizable(width=FALSE, height=FALSE)

        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)   
        self.grid_rowconfigure(1, weight=1)   
        self.grid_rowconfigure(2, weight=1)   

        
        self.ChatLog = customtkinter.CTkTextbox(master=self, font=("Arial", 10), wrap="word")
        self.ChatLog.configure(state=NORMAL)
        self.ChatLog.grid(row=0, column=0, padx=20, pady=5, sticky="ew", columnspan=2)

        
        self.ChatLog.tag_config("userinput", background="deep sky blue", foreground="black", justify="right")
        self.ChatLog.tag_config("botresponse", background="yellow", foreground="black", justify="left")
        self.ChatLog.tag_config("timestamp", foreground="red", justify="right")

        self.ChatLog.insert(END,textwrap.fill(f"QuinSpark Chat Bot Ready....",100))
        self.ChatLog.insert(END,'\n')
        self.ChatLog.configure(state=DISABLED)

        
        self.EntryBox = customtkinter.CTkTextbox(self, font=("Arial",10), wrap="word", fg_color="indian red", text_color="black")
        self.EntryBox.grid(row=1, column=0, padx=20, pady=5, sticky="nsw", columnspan=1)

        
        self.Hint =  customtkinter.CTkTextbox(master=self, font=("Arial",14), wrap="word", fg_color="indian red", text_color="black")
        self.Hint.insert("1.0", "Please click in the Chat Box and provide your medical symptoms for diagnosis....")
        self.Hint.grid(row=1, column=0, padx=20, pady=5, sticky="nsw", columnspan=1)

        
        self.control_frame = customtkinter.CTkFrame(self)
        self.control_frame.grid(row=1, column=1, padx=(0,20), pady=(5,5), sticky="nsew", columnspan=2)

        
        self.SendButton = customtkinter.CTkButton(master=self.control_frame, text="Send", command=self.button_send)
        self.SendButton.grid(row=0, column=0, padx=(10,30), pady=(10,10), sticky="ew", columnspan=1)

        
        self.ResetButton = customtkinter.CTkButton(master=self.control_frame, text="Reset", command=self.button_reset)
        self.ResetButton.grid(row=1, column=0, padx=(10,30), pady=(10,10), sticky="ew", columnspan=1)

        self.ProgressBar = customtkinter.CTkProgressBar(self, orientation="horizontal")
        self.ProgressBar.grid(row=2, column=0, padx=20, pady=(0,5), sticky="ew", columnspan=2)
        self.ProgressBar.set(0)

        
        self.Hint.bind("<FocusIn>", self.remove_hint)
        
        self.EntryBox.bind("<FocusOut>", self.add_hint)

        
        self.bind('<Return>', self.send)
        self.update()

    def remove_hint(self, event):
        """Hides the Data Entry Text Box by adding a Hint for Data Entry"""
        self.Hint.grid_forget()
        self.EntryBox.focus_set()

    def add_hint(self, event):
        """Dynamically replaces the Data Entry Text Box with a Placeholder Hint"""
        if self.hint_flag == 1:
            self.Hint.grid(row=1, column=0, padx=20, pady=5, sticky="nsw", columnspan=1)

    def button_reset(self):
        """Resets the Application"""
        self.ChatLog.configure(state=NORMAL)
        self.ChatLog.delete('1.0', END)
        self.ChatLog.configure(state=DISABLED)
        self.ProgressBar.stop()
        self.ProgressBar.set(0)

    def send(self, event):
        """Sends the Message to the ChatBot"""
        self.button_send()

    def button_send(self):
        """Extracts the message from the textbox, updates the Transcript, and clears the data entry form."""

        
        getmsg = self.EntryBox.get("1.0", 'end-1c').strip()
        msg = textwrap.fill(getmsg,250)
        self.EntryBox.delete("0.0", END)

        
        if len(msg) > 0:

            self.ProgressBar.set(10)
            self.ProgressBar.start()

            
            self.ChatLog.configure(state=NORMAL)

            
            self.ChatLog.insert(END, self.get_current_time(), ("timestamp"))
            self.ChatLog.insert(END, "\n" + getmsg + "\n\n", ("userinput"))

            
            self.ProgressBar.set(20)

            
            response = self.engine.user_entry(msg)

            
            self.ProgressBar.set(75)

            
            self.ChatLog.insert(END,"\n" + textwrap.fill(response)+'\n\n', ("botresponse"))

            
            self.ChatLog.configure(state=DISABLED)
            self.ChatLog.yview(END)

            
            self.ProgressBar.set(100)
            self.ProgressBar.stop()

    def update(self):
        """Runs automatically and modifies the send button if data is available"""

        
        if self.EntryBox.get("1.0", 'end-1c').strip() == '':
            self.SendButton['state'] = DISABLED
            self.hint_flag = 1

        
        elif self.EntryBox.get("1.0", 'end-1c').strip() != '':
            self.SendButton['state'] = ACTIVE
            self.hint_flag = 0

        
        self.after(100, self.update)

    @staticmethod
    def get_current_time() -> str:
        """Returns a Date/Time Formatted String"""
        now = datetime.now()
        return now.strftime("%D - %H:%M:%S \n")


if __name__ == "__main__":
    customtkinter.set_appearance_mode("light")

    
    engine = ChatBotEngine()

    
    app = ChatBotApp(engine)

    
    app.mainloop()