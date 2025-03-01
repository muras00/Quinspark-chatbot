import os
import folium
import requests
from geopy.geocoders import Nominatim
from selenium import webdriver
import time


def get_emergency_services(location_name, radius=5000):
    geolocator = Nominatim(user_agent="chatbot_emergency_map")
    location = geolocator.geocode(location_name + ", England")
    if not location:
        return None, "Sorry, I couldn't find the location."

    latitude, longitude = location.latitude, location.longitude
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
        [out:json];
        node
          [amenity~"hospital|police|fire_station"]
          (around:{radius},{latitude},{longitude});
        out body;
    """
    response = requests.get(overpass_url, params={"data": overpass_query})
    data = response.json()
    return data, (latitude, longitude)


def generate_map(location_name, output_file="emergency_map.html"):
    data, center_coords = get_emergency_services(location_name)
    if not data:
        return None, "No emergency services found."

   
    map_object = folium.Map(location=center_coords, zoom_start=12)
    folium.Marker(center_coords, popup="Center Location").add_to(map_object)


    for element in data.get("elements", []):
        lat, lon = element["lat"], element["lon"]
        name = element.get("tags", {}).get("name", "Unnamed")
        amenity = element.get("tags", {}).get("amenity", "Service")
        folium.Marker(
            [lat, lon],
            popup=f"{name} ({amenity})",
            icon=folium.Icon(color="red")
        ).add_to(map_object)


    map_object.save(output_file)
    return output_file


def save_map_as_image(html_file, image_file="map.png"):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("window-size=1024,768")

    driver = webdriver.Chrome(options=options)
    driver.get("file://" + os.path.abspath(html_file))
    time.sleep(2)
    driver.save_screenshot(image_file)
    driver.quit()
    return image_file


def load_emergency_services_map(location):
    html_file = generate_map(location)
    if not html_file:
        return "Sorry, no services found."
    image_file = save_map_as_image(html_file)
    return image_file
