import os
import folium
import time
from geopy.geocoders import Nominatim
from selenium import webdriver
from PIL import Image
import customtkinter



def generate_map(location_name, output_html="map.html"):
    geolocator = Nominatim(user_agent="chatbot_england_map")
    location = geolocator.geocode(location_name + ", England")

    if not location:
        return None, "Sorry, I couldn't find the location."

    
    map_object = folium.Map(location=[location.latitude, location.longitude], zoom_start=12)
    folium.Marker([location.latitude, location.longitude], popup=location_name).add_to(map_object)

   
    map_object.save(output_html)
    return output_html, f"Here is the map for {location_name}."



def save_map_as_image(html_file, output_image="map.png"):
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("window-size=1024,768")

    driver = webdriver.Chrome(options=options)  
    driver.get("file://" + os.path.abspath(html_file))

  
    time.sleep(2)

    
    driver.save_screenshot(output_image)
    driver.quit()
    return output_image



class ChatBotApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("England Map Chatbot")
        self.geometry("600x600")
        self.resizable(False, False)

        
        self.chat_log = customtkinter.CTkTextbox(self, wrap="word", font=("Arial", 12))
        self.chat_log.pack(padx=10, pady=10, fill="both", expand=True)
        self.chat_log.insert("1.0", "Welcome to the England Map Chatbot!\nType a location to generate its map.\n\n")

       
        self.user_input = customtkinter.CTkEntry(self, placeholder_text="Type a location in England...")
        self.user_input.pack(padx=10, pady=10, fill="x")
        self.user_input.bind("<Return>", self.handle_input)

       
        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.handle_input)
        self.send_button.pack(padx=10, pady=10)

        
        self.image_label = customtkinter.CTkLabel(self, text="Map will be displayed here.")
        self.image_label.pack(pady=10)

    
    def handle_input(self, event=None):
        user_message = self.user_input.get().strip()
        if not user_message:
            return

       
        self.chat_log.insert("end", f"You: {user_message}\n")
        self.user_input.delete(0, "end")

        
        self.respond_with_map(user_message)

    
    def respond_with_map(self, location_name):
        self.chat_log.insert("end", "Bot: Generating map, please wait...\n")

        
        html_file, response_message = generate_map(location_name)
        if not html_file:
            self.chat_log.insert("end", f"Bot: {response_message}\n")
            return

        
        image_file = save_map_as_image(html_file)

        
        self.chat_log.insert("end", f"Bot: {response_message}\n")

        
        map_image = customtkinter.CTkImage(light_image=Image.open(image_file), size=(500, 300))
        self.image_label.configure(image=map_image, text="")
        self.image_label.image = map_image  



if __name__ == "__main__":
    app = ChatBotApp()
    app.mainloop()
