import flickrapi
import requests
import os
from PIL import Image
from bs4 import BeautifulSoup

api_key = "fa47d1618c96629830cad08efdb49875"
api_secret = "a008ca98a9500197"

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
# Directory where the downloaded photos will be saved
download_dir = 'lego_images/'

# Connect to the Flickr API

# Search for photos matching the specified criteria
def add_legoset(search_terms: list[str]):
    b = 9100
    for term in search_terms:
        for i in [1,2,3]:
            print(i)
            photos = flickr.photos.search(
                #text='lego set',
                tags=term,
                color_codes="c",
                sort="relevance",
                per_page=100,
                page=i,
                extras='url_c'
            )

            # Download each photo and save it to the download directory
            
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


#add_legoset(["lego bricks"])
