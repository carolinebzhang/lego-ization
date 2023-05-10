import flickrapi
import requests
import os
from PIL import Image
from bs4 import BeautifulSoup

api_key = 'REMOVED'
api_secret = 'REMOVED'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

# Search for photos matching the specified criteria
def add_images(search_terms: list[str], download_dir: str):
    b = 9100
    for term in search_terms:
        for i in range(3):
            photos = flickr.photos.search(
                tags=term,
                #color_codes="c", 
                sort="relevance",
                per_page=100,
                page=i,
                extras='url_c'
            )
            
            for photo in photos['photos']['photo']:
                
                try:
                    url = photo['url_c']
                    new_path = str(b)
                    file_path = new_path + ".jpg"
                    response = requests.get(url)
                    filename = os.path.join(download_dir, file_path)
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Saved {filename}")
                    b+=1
                except Exception as e:
                    continue


add_images(["scene", "landscape", "houses"], "scene_images")
